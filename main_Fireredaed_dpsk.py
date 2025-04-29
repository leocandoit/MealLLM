#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音识别和智能分析处理程序

该程序实现了以下功能：
1. 从指定文件夹读取音频文件
2. 使用FunASR进行语音识别(ASR)
3. 通过大语言模型(LLM)分析文本中的外卖意图和食物信息
4. 将结果保存到输出文件中
"""

# 导入必要的库
import os
import asyncio
import time
import json,random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from torch import nn
import re
import json
import sys
# 语音识别相关库
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
# LLM API相关库
# from openai import AsyncOpenAI
# from dotenv import load_dotenv
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from fireredasr.models.fireredasr import FireRedAsr  # 导入新的 FireRedASR
# 加载环境变量(.env文件)
# load_dotenv()

# 配置日志
# 创建 logs 目录，避免路径不存在
Path("./logs").mkdir(parents=True, exist_ok=True)

# 日志格式
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台Handler（输出到屏幕）
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# 文件Handler（保存到文件）
file_handler = logging.FileHandler("./logs/output.log", mode='a', encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# 把Handler加到logger上
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 让openai和httpx的日志静音
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


#### 请提前到 魔搭社区下载ASR模型 #####
'''
下载方式： 在bash运行：
modelscope download --model iic/SenseVoiceSmall

'''

# 全局配置参数
CONFIG = {
    "audio_folder": "./project/convert_audio",  # 音频文件夹路径
    # "max_files": 30,                    # 最大处理文件数
    "output_dir": "./output",           # 输出目录
    "output_file": "output.txt",        # 结果文件名
    "uuid_file": "A.txt",               # UUID顺序文件
    # "llm_model": "deepseek-v3-250324",  # LLM模型名称
    # "llm_temperature": 0.7,             # LLM温度参数
    # "llm_timeout": 1800.0               # LLM超时时间(秒)
    "model_type": "aed",  # 这里指定模型类型
    "model_dir": "/home/disk_10T/xjq_data/waimaituijian/Datawhale_elm_zhihuiyanglao/FireRedASR/pretrained_models/FireRedASR-AED-L",  # 你的模型路径
    "use_gpu": True,
}

def load_asr_model() -> FireRedAsr:
    """
    加载新的 FireRedASR 语音识别模型
    
    返回:
        FireRedAsr: 加载好的 FireRedASR 模型实例
    """
    logger.info("正在加载FireRedASR模型...")
    model = FireRedAsr.from_pretrained(
        CONFIG["model_type"],
        CONFIG["model_dir"]
    )
    # 检测多GPU情况，并封装DataParallel
    if torch.cuda.device_count() > 1:
        logger.info(f"检测到 {torch.cuda.device_count()} 块GPU，使用DataParallel加速推理")
        model.model = torch.nn.DataParallel(model.model)
    else:
        logger.info("只检测到单块GPU，正常加载")

    logger.info("FireRedASR模型加载完成")
    return model

def find_audio_files(folder_path: str) -> tuple:
    """
    查找指定文件夹中的音频文件
    
    参数:
        folder_path: 音频文件夹路径
        
    返回:
        tuple: (文件路径列表, 任务名称列表)
    """
    logger.info(f"正在查找音频文件: {folder_path}")
    filename = []
    taskname = []
    
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            temp_name = os.path.join(root, file)
            filename.append(str(temp_name))
            taskname.append(file.split('.')[0])
    
    logger.info(f"找到 {len(filename)} 个音频文件")
    return filename, taskname

def process_audio_with_asr(model: FireRedAsr, audio_files: List[str]) -> List[str]:
    """
    使用新的 FireRedASR 模型处理音频文件（每500条一个batch）
    
    参数:
        model: FireRedASR模型
        audio_files: 音频文件路径列表
        
    返回:
        List[str]: 识别的文本列表
    """
    logger.info(f"正在处理 {len(audio_files)} 个音频文件...")

    batch_uttid = [os.path.splitext(os.path.basename(wav))[0] for wav in audio_files]
    batch_wav_path = audio_files

    asr_texts = []

    batch_size = 200  # 🔥 每500条音频做一次推理

    with open('asr_texts.log', 'w', encoding='utf-8') as f_log:
        for start_idx in range(0, len(batch_uttid), batch_size):
            end_idx = start_idx + batch_size
            sub_uttid = batch_uttid[start_idx:end_idx]
            sub_wav_path = batch_wav_path[start_idx:end_idx]

            logger.info(f"正在处理第 {start_idx} 到 {end_idx} 条音频...")

            # 调用 FireRedASR 推理
            results = model.transcribe(
                sub_uttid,
                sub_wav_path,
                {
                    "use_gpu": 1 if CONFIG["use_gpu"] else 0,
                    "beam_size": 3,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.25,
                    "aed_length_penalty": 0.6,
                    "eos_penalty": 1.0
                }
            )

            # 提取识别文本
            for item in results:
                full_text = item.get('text', '')
                if full_text.strip():
                    extracted_text = full_text.strip()
                    asr_texts.append(extracted_text)
                    f_log.write(extracted_text + '\n')

    logger.info(f"FireRedASR处理完成，提取了 {len(asr_texts)} 条文本")
    return asr_texts

def format_llm_prompt(user_input: str) -> str:
    """
    格式化LLM提示词
    
    参数:
        user_input: 用户输入文本
        
    返回:
        str: 格式化后的提示词
    """
    return f"""识别用户输入文本,返回满足要求的JSON。JSON必须包含两个字段: 'Call_elm' 和 'Food_candidate'。
'Call_elm': 布尔类型(boolean)，表示用户是否明确表达了点外卖的意图 (true or false)。
'Food_candidate': 字符串类型(string)，表示用户想要点的具体食物名称。如果没有明确提到食物，则返回空字符串 ""。

严格按照JSON格式输出，不要包含任何额外的解释或标记。

例如输入：天猫精灵要一份大碗香浓皮蛋瘦肉粥。
输出：
{{
  "Call_elm": true,
  "Food_candidate": "大碗香浓皮蛋瘦肉粥"
}}

例如输入：今天天气怎么样？
输出：
{{
  "Call_elm": false,
  "Food_candidate": ""
}}

当前用户输入：{user_input}

请输出JSON:"""

# 本地模型路径，替换成你的 DeepSeek 7B 模型地址
model_name = "/home/disk_10T/xjq_data/waimaituijian/Datawhale_elm_zhihuiyanglao/deepseek-llm-7b-chat"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 推荐 deepseek 用 bf16
    device_map="auto"
)

# 加载生成配置，并设置 pad_token_id
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 封装同步调用函数
def call_deepseek_local(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    # 注意 DeepSeek7B 默认就是 user-assistant 风格，不需要 system prompt
    input_tensor = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_tensor = input_tensor.to(model.device)

    outputs = model.generate(
        input_tensor.to(model.device),
        max_new_tokens=100
    )

    # 只取新生成的部分
    generated_ids = outputs[0][input_tensor.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

# 封装异步调用函数
async def call_llm_async(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_deepseek_local, prompt)


async def process_texts_with_llm(texts: List[str]) -> List[Dict]:
    if not texts:
        logger.warning("没有文本需要处理")
        return []
    
    num_texts = len(texts)
    prompts = [format_llm_prompt(text) for text in texts]
     
    logger.info(f"正在处理 {num_texts} 条文本...")
    start_time = time.perf_counter()

    tasks = [call_llm_async(prompt=p) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f"LLM处理完成，耗时 {total_time:.2f} 秒")

    processed_results = []
    successful_calls = 0
    failed_calls = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({"error": f"{type(result).__name__}: {result}"})
            failed_calls += 1
        else:
            # 解析JSON结果
            try:
                # 移除可能的Markdown代码块标记
                if isinstance(result, str):
                    if result.startswith("```json\n") and result.endswith("\n```"):
                        result = result[7:-4].strip()
                    elif result.startswith("```") and result.endswith("```"):
                        result = result[3:-3].strip()
                
                # 解析JSON
                parsed_json = json.loads(result)
                processed_results.append(parsed_json)
                successful_calls += 1
            except json.JSONDecodeError as json_err:
                # JSON解析错误
                logger.error(f"JSON解析错误 #{i+1}: {json_err}. 内容: '{result[:50]}...'")
                processed_results.append({"error": "Invalid JSON", "content": result})
                failed_calls += 1
            except Exception as parse_err:
                # 其他处理错误
                logger.error(f"结果处理错误 #{i+1}: {parse_err}")
                processed_results.append({"error": f"处理错误: {parse_err}", "content": result})
                failed_calls += 1

    logger.info(f"结果统计: {successful_calls} 成功, {failed_calls} 失败")
    return processed_results


def read_uuid_order(file_path: str) -> List[str]:
    """
    从A.txt文件读取UUID顺序
    
    参数:
        file_path: A.txt文件路径
        
    返回:
        List[str]: UUID列表，按照文件中的顺序
    """
    uuids = []
    try:
        logger.info(f"正在读取UUID顺序文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            next(f)  
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 0 and parts[0]:  # 确保UUID不为空
                    uuids.append(parts[0])
        
        logger.info(f"读取了 {len(uuids)} 个UUID")
        return uuids
    except Exception as e:
        logger.error(f"读取UUID顺序文件失败: {e}")
        return []


def format_and_save_results(tasknames: List[str], asr_texts: List[str], 
                           llm_results: List[Dict], output_file: str = None,
                           uuid_order: List[str] = None) -> None:
    """
    格式化并保存结果，可以按照指定的UUID顺序
    
    参数:
        tasknames: 任务名称列表
        asr_texts: ASR文本列表
        llm_results: LLM处理结果列表
        output_file: 输出文件路径（可选）
        uuid_order: UUID顺序列表（可选）
    """
    if output_file is None:
        output_file = CONFIG["output_file"]
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备结果数据
    results_data = []
    for i in range(min(len(tasknames), len(asr_texts), len(llm_results))):
        results_data.append({
            "taskname": tasknames[i],
            "asr_text": asr_texts[i],
            "call_elm": int(bool(llm_results[i].get('Call_elm'))) if isinstance(llm_results[i], dict) else 'ERROR',
            "food_candidate": llm_results[i].get('Food_candidate', '') if isinstance(llm_results[i], dict) else 'ERROR'
        })
    
    # 如果提供了UUID顺序，按照UUID顺序重新排列结果
    if uuid_order and len(uuid_order) > 0:
        logger.info("根据UUID顺序重新排列结果...")
        
        # 创建任务名称到结果的映射
        taskname_to_result = {item["taskname"]: item for item in results_data}
        
        # 按照UUID顺序重新排列结果
        ordered_results = []
        missing_uuids = []
        for uuid in uuid_order:
            if uuid in taskname_to_result:
                ordered_results.append(taskname_to_result[uuid])
            else:
                missing_uuids.append(uuid)
                # 为缺失的UUID创建一个默认结果
                ordered_results.append({
                    "taskname": uuid,
                    "asr_text": "未处理",
                    "call_elm": 0,
                    "food_candidate": ""
                })
        
        if missing_uuids:
            logger.warning(f"有 {len(missing_uuids)} 个UUID在处理结果中找不到: {missing_uuids[:5]}...")
        
        # 替换原始结果数据
        results_data = ordered_results
        logger.info(f"结果已按UUID顺序重新排列，共 {len(results_data)} 条")
    
    # 格式化结果为文本
    datas = [
        f"{item['taskname']}\t{item['asr_text']}\t{item['call_elm']}\t{item['food_candidate']}"
        for item in results_data
    ]
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(datas))
    
    logger.info(f"结果已保存到文件: {output_file}")


def generate_test_asr_texts(count: int) -> List[str]:
    """
    生成固定的测试ASR文本数据
    
    参数:
        count: 需要生成的数据数量
        
    返回:
        List[str]: 测试ASR文本列表
    """
    # 基础测试文本列表
    test_texts = [
        "天猫精灵要一份大碗香浓皮蛋瘦肉粥。",
        "我想听周杰伦的歌。",
        "天猫精灵来份南瓜小米粥。",
        "今天天气怎么样？",
        "天猫精灵来份萝卜炖排骨汤。"
    ]
    
    # 循环使用基础文本来生成所需数量的数据
    result = []
    for i in range(count):
        result.append(test_texts[i % len(test_texts)])
    
    return result


def generate_test_llm_results(count: int) -> List[Dict]:
    """
    生成固定的测试LLM结果数据
    
    参数:
        count: 需要生成的数据数量
        
    返回:
        List[Dict]: 测试LLM结果列表
    """
    # 基础测试结果列表（与测试文本对应）
    test_results = [
        {"Call_elm": True, "Food_candidate": "大碗香浓皮蛋瘦肉粥"},
        {"Call_elm": False, "Food_candidate": ""},
        {"Call_elm": True, "Food_candidate": "南瓜小米粥"},
        {"Call_elm": False, "Food_candidate": ""},
        {"Call_elm": True, "Food_candidate": "萝卜炖排骨汤"}
    ]
    
    # 循环使用基础结果来生成所需数量的数据
    result = []
    for i in range(count):
        result.append(test_results[i % len(test_results)])
    
    return result


async def main_process() -> None:
    """主处理流程"""
    logger.info("===== 开始处理流程 =====")
    
    # 0. 读取UUID顺序（如果存在）
    uuid_order = None
    uuid_file = CONFIG.get("uuid_file")
    if uuid_file and os.path.exists(uuid_file):
        uuid_order = read_uuid_order(uuid_file)
    
    # 1. 查找音频文件
    filenames, tasknames = find_audio_files(CONFIG["audio_folder"])
    
    # 获取总文件数量
    total_files = len(filenames)
    logger.info(f"找到总共 {total_files} 个音频文件")
    
    # 限制处理文件数量随机（0-30）
    # normal_process_limit = random.randint(10, 30)
    # files_to_process_normally = min(normal_process_limit, total_files)
    files_to_process_normally = total_files
    # 处理前N个文件
    logger.info(f"将正常处理前 {files_to_process_normally} 个文件")
    
    # 2. 加载ASR模型
    asr_model = load_asr_model()
    
    # 3. 对前N个文件进行ASR处理
    if files_to_process_normally > 0:
        normal_filenames = filenames[:files_to_process_normally]
        normal_tasknames = tasknames[:files_to_process_normally]
        
        logger.info(f"开始对前 {files_to_process_normally} 个文件进行ASR处理...")
        asr_texts = process_audio_with_asr(asr_model, normal_filenames)
        logger.info(f"ASR处理完成，获得 {len(asr_texts)} 条文本")
        
        # 4. 对前N个文件的ASR结果进行LLM处理
        logger.info("开始对ASR结果进行LLM处理...")
        llm_results = await process_texts_with_llm(asr_texts)
        logger.info("LLM处理完成")
    else:
        # 如果没有文件需要正常处理，则创建空列表
        asr_texts = []
        llm_results = []
    # # 5. 处理剩余文件（使用固定测试数据）
    # if total_files > files_to_process_normally:
    #     remaining_count = total_files - files_to_process_normally
    #     logger.info(f"为剩余的 {remaining_count} 个文件生成测试数据...")
        
    #     # 生成固定的测试数据
    #     test_asr_texts = generate_test_asr_texts(remaining_count)
    #     test_llm_results = generate_test_llm_results(remaining_count)
        
    #     # 合并结果
    #     asr_texts.extend(test_asr_texts)
    #     llm_results.extend(test_llm_results)
        
    #     logger.info(f"测试数据生成完成，总共有 {len(asr_texts)} 条ASR文本和 {len(llm_results)} 条LLM结果")
    # 6. 保存结果（按UUID顺序）
    format_and_save_results(tasknames, asr_texts, llm_results, uuid_order=uuid_order)
    
    logger.info("===== 处理流程完成 =====")


if __name__ == "__main__":
    # 运行主流程
    asyncio.run(main_process())
