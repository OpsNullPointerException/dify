import re


class CleanProcessor:
    """文本清理处理器：负责清理文档中的无效字符和不需要的内容"""
    
    @classmethod
    def clean(cls, text: str, process_rule: dict) -> str:
        # 默认清理：移除控制字符和无效符号
        # 处理模型输出中的特殊标记符（防止与系统内部标记冲突）
        text = re.sub(r"<\|", "<", text)  # 将 <| 替换为 < （避免与ChatML等格式标记冲突）
        text = re.sub(r"\|>", ">", text)  # 将 |> 替换为 > （配对处理特殊标记）
        # 移除控制字符（除了换行、制表符等常用字符）
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]", "", text)
        # 移除 Unicode 字节顺序标记 U+FFFE
        text = re.sub("\ufffe", "", text)

        # 应用用户自定义的预处理规则
        rules = process_rule["rules"] if process_rule else {}
        if "pre_processing_rules" in rules:
            pre_processing_rules = rules["pre_processing_rules"]
            for pre_processing_rule in pre_processing_rules:
                # 规则1：移除多余空格和换行
                if pre_processing_rule["id"] == "remove_extra_spaces" and pre_processing_rule["enabled"] is True:
                    # 将3个或更多连续换行替换为2个换行
                    pattern = r"\n{3,}"
                    text = re.sub(pattern, "\n\n", text)
                    # 将多个连续空白字符（包括各种Unicode空格）替换为单个空格
                    pattern = r"[\t\f\r\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}"
                    text = re.sub(pattern, " ", text)
                    
                # 规则2：移除邮箱和URL（但保留Markdown图片链接）
                elif pre_processing_rule["id"] == "remove_urls_emails" and pre_processing_rule["enabled"] is True:
                    # 移除邮箱地址
                    pattern = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
                    text = re.sub(pattern, "", text)

                    # 智能处理URL：移除普通URL但保留Markdown图片链接
                    # 步骤1：临时替换Markdown图片URL为占位符
                    markdown_image_pattern = r"!\[.*?\]\((https?://[^\s)]+)\)"
                    placeholders: list[str] = []

                    def replace_with_placeholder(match, placeholders=placeholders):
                        url = match.group(1)
                        placeholder = f"__MARKDOWN_IMAGE_URL_{len(placeholders)}__"
                        placeholders.append(url)
                        return f"![image]({placeholder})"

                    text = re.sub(markdown_image_pattern, replace_with_placeholder, text)

                    # 步骤2：移除所有剩余的URL
                    url_pattern = r"https?://[^\s)]+"
                    text = re.sub(url_pattern, "", text)

                    # 步骤3：恢复Markdown图片URL
                    for i, url in enumerate(placeholders):
                        text = text.replace(f"__MARKDOWN_IMAGE_URL_{i}__", url)
        return text

    def filter_string(self, text):
        """保留方法：用于兼容性，目前直接返回原文本"""
        return text
