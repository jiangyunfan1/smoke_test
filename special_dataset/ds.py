import json
import os

def generate_txt_files_from_json(json_file_path, output_dir="output"):
    """
    从JSON文件生成txt文件
    
    参数:
        json_file_path: JSON文件路径
        output_dir: 输出目录，默认为"output"
    """
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：{json_file_path} 不是有效的JSON文件")
        return
    
    # 统计每个category的出现次数，用于生成序号
    category_count = {}
    
    # 处理每个条目
    for i, item in enumerate(data, 1):
        try:
            category = item["category"]
            content = item["content"]
            
            # 更新category计数
            if category not in category_count:
                category_count[category] = 1
            else:
                category_count[category] += 1
            
            # 生成文件名
            filename = f"{category}_{category_count[category]}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"已创建: {filename}")
            
        except KeyError as e:
            print(f"警告：第{i}个条目缺少必要的字段 {e}")
        except Exception as e:
            print(f"处理第{i}个条目时出错: {e}")
    
    print(f"\n完成！共生成 {len(data)} 个txt文件到目录: {output_dir}")

def main():
    # 配置参数
    json_file_path = "分词边界攻击.json"  # 修改为你的JSON文件路径
    output_dir = "case_12_10"      # 输出目录
    
    # 生成txt文件
    generate_txt_files_from_json(json_file_path, output_dir)

if __name__ == "__main__":
    main()