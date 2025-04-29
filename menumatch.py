from sentence_transformers import SentenceTransformer
import torch

# 配置
output_txt = "/home/disk_10T/xjq_data/waimaituijian/Datawhale_elm_zhihuiyanglao/output.txt"
menu_txt = "/home/disk_10T/xjq_data/waimaituijian/Datawhale_elm_zhihuiyanglao/project/dim_ai_exam_food_category_filter_out.txt"
output_txt_new = "/home/disk_10T/xjq_data/waimaituijian/Datawhale_elm_zhihuiyanglao/output_updated.txt"

# 加载中文embedding模型
embed_model = SentenceTransformer("/home/disk_10T/xjq_data/waimaituijian/Datawhale_elm_zhihuiyanglao/BAAI/bge-small-zh-v1.5")

# 读取菜单
menu_items = []
with open(menu_txt, 'r', encoding='utf-8') as f:
    for line in f:
        fields = line.strip().split('\t')
        if fields:
            menu_items.append(fields[0])

print(f"菜单一共加载了 {len(menu_items)} 个菜品")

# 菜单embedding
menu_embeddings = embed_model.encode(menu_items, convert_to_tensor=True)

# 处理 output.txt
new_lines = []
threshold = 0.7  # 匹配阈值

with open(output_txt, 'r', encoding='utf-8') as f:
    for line in f:
        fields = line.strip().split('\t')
        if len(fields) < 4:
             # 不足4列，自动补齐空列
            fields += [''] * (4 - len(fields))

        uuid = fields[0]
        text = fields[1]
        current_label = fields[2]
        food_candidate = fields[3]

        # 如果food_candidate为空，直接置0并清空
        if not food_candidate.strip():
            updated_line = f"{uuid}\t{text}\t0\t"
            new_lines.append(updated_line)
            continue

        # food_candidate不为空，继续语义匹配
        food_emb = embed_model.encode(food_candidate, convert_to_tensor=True)

        cos_sim = torch.nn.functional.cosine_similarity(food_emb.unsqueeze(0), menu_embeddings)

        max_sim, max_idx = torch.max(cos_sim, dim=0)  # 找最大相似度和对应index
        max_sim = max_sim.item()
        max_idx = max_idx.item()

        if max_sim >= threshold:
            # 匹配成功，置1，第四列替换成最相似菜单项
            new_label = '1'
            updated_food_candidate = menu_items[max_idx]  # 替换为最相似的菜单菜名
        else:
            # 匹配失败，置0，清空
            new_label = '0'
            updated_food_candidate = ""

        updated_line = f"{uuid}\t{text}\t{new_label}\t{updated_food_candidate}"
        new_lines.append(updated_line)

# 保存新的 output
with open(output_txt_new, 'w', encoding='utf-8') as f:
    for line in new_lines:
        f.write(line + '\n')

print(f"完成更新，保存到 {output_txt_new}")
