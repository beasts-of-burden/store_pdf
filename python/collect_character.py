import re
import json

# ---------- 示例文本 ----------
text = """
炭治郎
• 绿黑格子外套身前最下方都是绿色格子
• 使用格子设计时，注意格子不要太小太密集，另外边缘部分也尽量都保持正方形，不要裁剪成长方形
• 队服的纽扣是银色的，有3个
祢豆子
• 使用竹筒设计时，左右两边的缎带必须一起画出来
• 发梢的变色部分是有明确边界线的，不是渐变
• 头发上的蝴蝶结，两个头是从同一边出来
善逸
• 外套和护腿的变色部分是渐变色，头发和眉毛的变色部分有明确边界线
• 外套上的三角形都是等边三角形，方向全部朝上，并且均匀分布。使用三角形设计时也
要注意，并且边缘处的三角形都要完整，不要被裁掉一半
• 队服的纽扣是银色的，有3个
伊之助
• 野猪头套的耳朵外面是粉色的，里面是紫色的（颜色不一样）
• 拿下头套后，头发的变色部分是渐变色
• 自行绘制野猪头套的时候要尽量画得可爱一点，不能看起来吓人
义勇
• 队服的纽扣是金色的，有4个
• 原则上禁止笑容
忍
• 外套、护腿、发梢、瞳孔的变色部分都是渐变色
• 外套和发饰边缘的斑点都不是纯白色的
• 队服的纽扣是金色的，有3个
杏寿郎
• 发梢的变色部分有明确边界线
• 后脑勺有个小辫子
• 队服的纽扣是金色的，有4个
• 外套外侧的橙色部分是渐变色
天元
• 耳朵上有耳环
• 队服的纽扣是金色的，有4个
• 宝石的形状很容易画错，绘制时请仔细参考设定
无一郎
• 队服的纽扣是金色的，有3个
• 发梢的变色部分有明确边界线
蜜璃
• 队服的纽扣是金色的，有4个
• 发梢的变色部分是渐变色
小芭内
• 队服的纽扣是金色的，有3个
• 蛇需要尽量把鳞片也画出来
行冥
• 队服的纽扣是金色的，有1个
• 眼泪注意不要流得太厉害
实弥
• 伤疤需要还原
• 瞳孔周围得血管需要还原
• 瞳孔是椭圆形的
• 队服的纽扣是金色的，有1个
• 笑起来只能是坏笑，不允许纯量的笑容
无惨
• 禁止笑容
"""

# ---------- 解析文本 ----------
data = []
current_character = None
current_rules = []

for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    # 如果这一行没有 •，就认为是角色名
    if not line.startswith("•"):
        if current_character:
            data.append({"character": current_character, "rules": current_rules})
        current_character = line
        current_rules = []
    else:
        # 去掉前面的 • 和空格
        rule_text = line.lstrip("•").strip()
        current_rules.append(rule_text)

# 添加最后一个角色
if current_character:
    data.append({"character": current_character, "rules": current_rules})

# ---------- 输出 JSON ----------
with open("rules_character.json", "w", encoding="utf-8") as f_json:
    json.dump(data, f_json, ensure_ascii=False, indent=2)

# ---------- 打印结果 ----------
print(json.dumps(data, ensure_ascii=False, indent=2))
