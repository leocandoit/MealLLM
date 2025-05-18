#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音识别和智能分析处理程序

该程序实现了以下功能：
1. 从指定文件夹读取音频文件
2. 使用 FunASR SenseVoiceSmall 进行语音识别（ASR）
3. 通过本地 Qwen3-8B 大语言模型（非思考模式）分析文本中的外卖意图和食物信息
4. 将结果保存到输出文件中（UUID\tASR 文本\t意图\t菜品）
"""

import os
import sys
import asyncio
import time
import json
import logging
import random
import traceback
import torch
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
# ASR
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 确保 logs 目录存在
Path("./logs").mkdir(parents=True, exist_ok=True)

# 全局变量，只加载一次
_tokenizer = None
_model = None
_model_loading = False
_last_error = None

# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler(sys.stdout);   ch.setFormatter(fmt)
fh = logging.FileHandler("./logs/output.log", encoding="utf-8"); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# 全局配置
CONFIG = {
    "audio_folder": "autodl-tmp/A_audio",    # 音频目录
    "uuid_file": "autodl-tmp/智慧养老_label/A.txt",                         # 可选：UUID 列表（每行一个，无扩展名）
    "asr_model_dir": "autodl-tmp/SenseVoiceSmall",       # SenseVoiceSmall 模型 ID 或本地路径
    "llm_model_dir": "autodl-tmp/Qwen3-8B",  # 本地 Qwen3-8B 模型目录
    "output_dir": "./output",                     # 输出目录
    "output_file": "output.txt",                  # 输出文件名
    "max_files": 0,                               # ≤0 表示全部
    "llm_temperature": 0.7,                       # LLM 温度
}

def load_qwen_model(model_dir="autodl-tmp/Qwen3-8B"):
    """加载 Qwen 模型，使用 4 位量化以减少内存使用并提高兼容性"""
    global _tokenizer, _model, _model_loading, _last_error
    
    # 如果模型已加载，直接返回
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    
    # 如果正在加载，等待
    if _model_loading:
        wait_count = 0
        while _model_loading and wait_count < 60:
            time.sleep(1)
            wait_count += 1
        
        # 检查加载是否完成
        if _tokenizer is not None and _model is not None:
            return _tokenizer, _model
        elif _last_error is not None:
            raise RuntimeError(f"模型加载失败: {_last_error}")
        else:
            raise TimeoutError("等待模型加载超时")
    
    # 开始加载模型
    _model_loading = True
    try:
        logger.info(f"开始加载 Qwen 模型: {model_dir}")
        
        # 加载分词器
        _tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=False
        )
        logger.info("分词器加载完成")
        
        # 使用 4 位量化配置，比 8 位更节省内存且更稳定
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 使用 4 位量化而不是 8 位
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # 加载模型，使用更保守的设置
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,  # 明确指定 float16
            offload_folder="offload",
            offload_state_dict=True
        )
        logger.info("模型加载完成")
        
        # 修复 generation_config 中可能为列表的字段
        gen_cfg = _model.generation_config
        if isinstance(gen_cfg.pad_token_id, list):
            gen_cfg.pad_token_id = gen_cfg.pad_token_id[0] if gen_cfg.pad_token_id else gen_cfg.eos_token_id
            logger.info(f"修复 pad_token_id: {gen_cfg.pad_token_id}")
            
        for param in ("bos_token_id", "eos_token_id", "decoder_start_token_id"):
            val = getattr(gen_cfg, param, None)
            if isinstance(val, list):
                setattr(gen_cfg, param, val[0] if val else gen_cfg.eos_token_id)
                logger.info(f"修复 {param}: {getattr(gen_cfg, param)}")
                
        _model.generation_config = gen_cfg
        logger.info("模型配置修复完成")
        
        # 预热模型
        logger.info("预热模型...")
        try:
            with torch.inference_mode():
                warmup_text = "你好，请介绍一下自己"
                warmup_inputs = _tokenizer([warmup_text], return_tensors="pt").to(_model.device)
                _model.generate(**warmup_inputs, max_new_tokens=10)
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败，但继续使用: {e}")
        
        return _tokenizer, _model
        
    except Exception as e:
        _last_error = str(e)
        logger.error(f"模型加载失败: {e}")
        logger.error(traceback.format_exc())
        _tokenizer, _model = None, None
        raise
    finally:
        _model_loading = False


def call_qwen(prompt: str, max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20):
    """调用 Qwen 模型生成回复，使用全局已加载的模型和量化"""
    try:
        tokenizer, model = load_qwen_model()
        
        # 构造聊天输入
        messages = [{"role": "user", "content": prompt}]
        inp = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 确保输入不会太长
        if len(inp) > 4000:
            logger.warning(f"输入过长 ({len(inp)} 字符)，截断至 4000 字符")
            inp = inp[:4000]
        
        # 转换为张量并移至设备
        inputs = tokenizer([inp], return_tensors="pt")
        
        # 明确创建 attention_mask - 全部设为1，因为我们没有填充
        # 这解决了 pad_token_id 与 eos_token_id 相同的问题
        attention_mask = torch.ones_like(inputs["input_ids"])
        
        # 将输入移至设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        attention_mask = attention_mask.to(model.device)

        # 生成时开启混合精度和推理模式
        with torch.autocast(device_type="cuda", dtype=torch.float16), torch.inference_mode():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,  # 使用明确创建的 attention_mask
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=0.0,
            )
        
        # 解码并返回
        input_length = inputs["input_ids"].shape[1]
        text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True)
        return text
        
    except Exception as e:
        logger.error(f"模型调用失败: {e}")
        logger.error(traceback.format_exc())
        # 返回错误信息的 JSON 字符串
        import json
        return json.dumps({"Call_elm": False, "error": str(e)}, ensure_ascii=False)

def init_qwen_model():
    """初始化 Qwen 模型"""
    logger.info("初始化 Qwen 模型...")
    try:
        # 加载模型
        load_qwen_model(CONFIG["llm_model_dir"])
        logger.info("Qwen 模型初始化完成")
        return True
    except Exception as e:
        logger.error(f"Qwen 模型初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_asr_model() -> AutoModel:
    """加载 FunASR SenseVoiceSmall 模型"""
    logger.info("加载 FunASR SenseVoiceSmall 模型...")
    model = AutoModel(
        model=CONFIG["asr_model_dir"],
        trust_remote_code=True,
        # 移除 remote_code 参数
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    logger.info("ASR 模型加载完成")
    return model

# 添加加载模型的代码
load_qwen_model(CONFIG["llm_model_dir"])

def call_qwen_local(prompt: str) -> str:
    """同步调用 Qwen3-8B 本地模型"""
    try:
        # 使用导入的 call_qwen 函数
        return call_qwen(prompt)
    except Exception as e:
        logger.error(f"LLM 调用异常: {e}")
        return json.dumps({"Call_elm": False, "error": str(e)})


async def process_texts_with_llm(texts: List[str]) -> List[Dict[str, Any]]:
    """并发异步调用 Qwen3-8B，返回解析后的 JSON 列表"""
    if not texts:
        return []
    logger.info(f"开始 LLM 处理 {len(texts)} 条文本")
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    
    # 使用更清晰的提示词格式
    prompts = []
    for txt in texts:
        prompt = f"""请分析以下文本，判断是否有叫外卖的意图。
如果有，返回JSON格式：{{"Call_elm": true, "Food_candidate": "食物名称"}}
如果没有，返回{{"Call_elm": false}}
严格按照JSON格式输出，不要包含任何额外的解释或标记。

文本：{txt}"""
        prompts.append(prompt)
    
    tasks = [loop.run_in_executor(None, call_qwen_local, p) for p in prompts]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results, suc, fail = [], 0, 0
    for i, r in enumerate(raw):
        try:
            # 处理异常情况
            if isinstance(r, Exception):
                logger.error(f"第{i}条 LLM 调用出错: {r}")
                results.append({"Call_elm": False, "error": str(r)})
                fail += 1
                continue
                
            # 确保r是字符串类型
            if not isinstance(r, str):
                logger.warning(f"第{i}条返回非字符串: {type(r)}")
                r = str(r)
            
            # 尝试解析JSON
            try:
                # 如果返回的是有效JSON字符串，直接解析
                parsed = json.loads(r)
                
                # 检查是否包含错误信息
                if "error" in parsed:
                    logger.warning(f"第{i}条返回错误: {parsed['error']}")
                    fail += 1
                else:
                    suc += 1
                    
                results.append(parsed)
            except json.JSONDecodeError as e:
                # 如果不是有效JSON，尝试从文本中提取意图
                logger.warning(f"第{i}条JSON解析失败: {e}, 原始文本: {r[:100]}...")
                
                # 简单启发式判断
                has_intent = "true" in r.lower() and "food" in r.lower()
                food_match = re.search(r'"Food_candidate"\s*:\s*"([^"]+)"', r)
                food = food_match.group(1) if food_match else ""
                
                results.append({"Call_elm": has_intent, "Food_candidate": food})
                suc += 1
        except Exception as e:
            logger.error(f"第{i}条处理异常: {e}")
            results.append({"Call_elm": False, "error": str(e)})
            fail += 1
            
    logger.info(f"LLM 完成，耗时 {time.perf_counter()-t0:.2f}s，成功{suc} 失败{fail}")
    logger.info(f"批次大小: {len(texts)}, 内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    return results

def find_audio_files(folder: str, uuid_file: str) -> (List[str], List[str]):
    """
    查找 WAV 音频：
    - 若 uuid_file 存在，按列表顺序加载 <UUID>.wav
    - 否则遍历目录
    返回：文件路径列表、对应的 UUID 列表
    """
    paths, uuids = [], []
    if uuid_file and os.path.isfile(uuid_file):
        logger.info(f"按 UUID 列表读取: {uuid_file}")
        with open(uuid_file, encoding="utf-8") as f:
            for line in f:
                uid = line.strip()
                if not uid: continue
                p = os.path.join(folder, f"{uid}.wav")
                if os.path.isfile(p):
                    paths.append(p); uuids.append(uid)
                else:
                    logger.warning(f"未找到: {p}")
    else:
        logger.info(f"遍历目录加载 WAV: {folder}")
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".wav"):
                    p = os.path.join(root, fn)
                    paths.append(p); uuids.append(Path(fn).stem)
    logger.info(f"发现 {len(paths)} 个音频文件")
    return paths, uuids

def process_audio_with_asr(model: AutoModel, files: List[str]) -> List[str]:
    """批量调用 ASR，返回转写文本列表"""
    logger.info(f"开始 ASR 识别 {len(files)} 个文件")
    t0 = time.perf_counter()
    
    # 添加健壮性处理
    results = []
    processed_files = []
    
    # 分批处理，每批最多10个文件
    batch_size = 10
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        logger.info(f"处理ASR批次 {i//batch_size + 1}/{(len(files)+batch_size-1)//batch_size}, 文件数: {len(batch_files)}")
        
        try:
            # 确保模型内部的CMVN数据是有效的NumPy数组
            if hasattr(model, 'frontend') and hasattr(model.frontend, 'cmvn'):
                model.frontend.cmvn = process_batch(model.frontend.cmvn, ensure_numpy=True)
                
            batch_results = model.generate(
                input=batch_files,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            results.extend(batch_results)
            processed_files.extend(batch_files)
        except Exception as e:
            logger.error(f"批次处理失败: {e}")
            # 尝试逐个处理文件
            for file in batch_files:
                try:
                    logger.info(f"单独处理文件: {os.path.basename(file)}")
                    single_result = model.generate(
                        input=[file],
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=True,
                        merge_length_s=15,
                    )
                    results.extend(single_result)
                    processed_files.append(file)
                except Exception as file_e:
                    logger.error(f"文件处理失败 {os.path.basename(file)}: {file_e}")
                    # 添加空结果占位
                    results.append({"text": f"[ASR失败: {os.path.basename(file)}]"})
    
    # 处理结果
    texts = []
    for res in results:
        try:
            text = rich_transcription_postprocess(res.get("text", ""))
            texts.append(text)
        except Exception as e:
            logger.error(f"后处理失败: {e}")
            texts.append(f"[后处理失败]")
    
    logger.info(f"ASR 完成，耗时 {time.perf_counter()-t0:.2f}s，处理 {len(processed_files)}/{len(files)} 个文件")
    return texts

def format_and_save_results(uuids: List[str],
                           asr_texts: List[str],
                           llm_results: List[Dict[str, Any]]):
    """格式化为 UUID\tASR\tINTENT\tFOOD 并保存"""
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    out_path = Path(CONFIG["output_dir"]) / CONFIG["output_file"]
    
    # 准备结果数据
    results_data = []
    for i in range(min(len(uuids), len(asr_texts), len(llm_results))):
        # 清理ASR文本中的表情符号
        clean_asr_text = clean_text(asr_texts[i])
        # 安全地检查 Call_elm 字段
        call = 0
        food = ""
        try:
            if isinstance(llm_results[i], dict):
                if llm_results[i].get("Call_elm") in [True, 1, "1", "true", "True"]:
                    call = 1
                    food = str(llm_results[i].get("Food_candidate", ""))
        except Exception as e:
            logger.warning(f"处理结果异常: {e}, res={llm_results[i]}")
        
        results_data.append({
            "uuid": uuids[i],
            "asr_text": clean_asr_text,  # 使用清理后的文本
            "call_elm": call,
            "food_candidate": food
        })
    
    # 格式化结果为文本
    lines = [
        f"{item['uuid']}\t{item['asr_text']}\t{item['call_elm']}\t{item['food_candidate']}"
        for item in results_data
    ]
    
    # 保存结果
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"结果已保存到 {out_path}")


def test_qwen_model():
    """测试 Qwen 模型是否正常工作"""
    logger.info("测试 Qwen 模型...")
    
    # 使用“仅返回 JSON”提示，确保模型返回纯粹的 JSON 格式
    test_prompt = "请仅返回 JSON 格式回答：今天天气怎么样？"
    
    try:
        # 使用 call_qwen 函数获取模型的返回结果
        result = call_qwen(test_prompt)
        logger.info(f"测试结果 (原始输出): {result}")  # 打印完整的模型返回内容
        
        # 清洗模型返回的文本，提取 JSON 部分
        clean_result = extract_json(result)
        logger.info(f"清洗后的 JSON 部分: {clean_result}")  # 打印清洗后的 JSON 部分
        
        # 尝试解析返回的结果，确保它是有效的 JSON
        json.loads(clean_result)
        logger.info("模型测试成功：返回了有效的 JSON")
        return True
    except json.JSONDecodeError as e:
        # 如果返回的结果不是有效的 JSON，则记录错误
        logger.error(f"模型测试失败：返回结果不是有效的 JSON，解析失败。错误信息: {e}")
        logger.error(f"清洗后的 JSON 内容: {clean_result}")  # 打印清洗后的内容
        return False
    except Exception as e:
        # 捕获其他异常，记录错误信息
        logger.error(f"模型测试失败: {e}")
        return False

def extract_json(text: str) -> str:
    """
    清洗返回结果：去除非 JSON 部分，只保留纯 JSON 内容
    """
    # 尝试提取 JSON 部分，忽略其他文本
    m = re.search(r"\{.*\}", text, re.DOTALL)  # 使用正则匹配 JSON
    if m:
        return m.group(0).strip()  # 返回匹配的 JSON 内容
    else:
        # 如果没有找到有效的 JSON，返回空字符串
        logger.warning("未能提取有效的 JSON 内容")
        return ""


async def main():
    logger.info("===== 开始主流程 =====")
    
    # 测试模型
    if not test_qwen_model():
        logger.error("模型测试失败，程序终止")
        return
    
    try:
        # 查找音频文件
        files, uuids = find_audio_files(CONFIG["audio_folder"], CONFIG["uuid_file"])
        total = len(files)
        if total == 0:
            logger.error("未找到任何音频文件，程序终止")
            return
            
        n = total if CONFIG["max_files"] <= 0 else min(CONFIG["max_files"], total)
        logger.info(f"处理前 {n}/{total} 个文件")
        sel_files = files[:n]; sel_uuids = uuids[:n]

        # 加载ASR模型并处理音频
        asr_model = load_asr_model()
        asr_texts = process_audio_with_asr(asr_model, sel_files)
        
        # 分批处理LLM
        batch_size = 400  
        all_llm_results = []
        for i in range(0, len(asr_texts), batch_size):
            batch_texts = asr_texts[i:i+batch_size]
            logger.info(f"处理LLM批次 {i//batch_size + 1}/{(len(asr_texts)+batch_size-1)//batch_size}")
            batch_results = await process_texts_with_llm(batch_texts)
            all_llm_results.extend(batch_results)
            
            # 添加短暂休息，让GPU有时间冷却
            if i + batch_size < len(asr_texts):
                await asyncio.sleep(1)
        
        # 保存结果
        format_and_save_results(sel_uuids, asr_texts, all_llm_results)
        
    except Exception as e:
        logger.error(f"主流程异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("===== 主流程完成 =====")


if __name__ == "__main__":
    # 首先测试模型
    if not test_qwen_model():
        logger.error("Qwen 模型测试失败，程序终止")
        sys.exit(1)
    
    # 如果测试通过，继续执行主程序
    asyncio.run(main())
