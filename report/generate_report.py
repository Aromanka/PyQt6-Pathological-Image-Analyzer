from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
import cv2
import numpy as np
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

simsun = "simsun"
pdfmetrics.registerFont(TTFont(simsun, "simsun.ttc"))

# ====================
# 模拟数据准备
# ====================

# 生成模拟病理图像（实际使用时替换为真实图像路径）
IMAGE_PATH = r'./image.png'
PDF_PATH = r'report.pdf'

# ====================
# PDF生成函数
# ====================

def create_report(report_info, cur_dir=r'./'):
    # load data
    # 患者信息
    patient_info = {
        "姓名": report_info["patient_name"],
        "年龄": "{}".format(report_info["patient_age"]),
        "性别": report_info["patient_gender"],
        "患者ID": report_info["patient_id"],
        "报告日期": datetime.now().strftime("%Y-%m-%d")
    }
    # 图像超参数
    width, height = report_info['image_resolution']
    image_params = {
        "图片名称": report_info["image_name"],
        "分辨率": f"{width}x{height}"
    }

    # 汇管区统计信息
    statistics_data = [
        ["指标", "数值"],
        ["汇管区总数", "{}".format(report_info["portal_num"])],
        ["正常汇管区总数", "{}".format(report_info["normal_portal_num"])],
        ["纤维化汇管区总数", "{}".format(report_info["fibre_portal_num"])],
        ["汇管区占比", "{}%".format(report_info["portal_area"])],
        ["正常汇管区占比", "{}%".format(report_info["normal_portal_area"])],
        ["纤维化汇管区占比", "{}%".format(report_info["fibre_portal_area"])]
    ]



    # 创建文档模板
    doc = SimpleDocTemplate(cur_dir+PDF_PATH,
                            pagesize=A4,
                            leftMargin=2 * cm,
                            rightMargin=2 * cm,
                            topMargin=1 * cm,
                            bottomMargin=1 * cm)

    # 样式定义
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="Title",
        fontSize=18,
        leading=24,
        alignment=1,  # 居中
        fontName=simsun
    )
    section_style = ParagraphStyle(
        name="Section",
        fontSize=12,
        leading=16,
        fontName=simsun,
        textColor=colors.darkblue
    )

    # 内容元素列表
    elements = []

    # 1. 标题
    elements.append(Paragraph("肝脏病理检查报告", title_style))
    elements.append(Spacer(1, 1 * cm))

    # 2. 患者信息表格
    patient_rows = [[k, v] for k, v in patient_info.items()]
    patient_table = Table(patient_rows, colWidths=[6 * cm, 10 * cm])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), simsun),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 1 * cm))

    # 3. 图像与参数
    # 图像处理
    img = Image(cur_dir+IMAGE_PATH, width=12 * cm, height=8 * cm)

    # 参数表格
    param_rows = [[k, v] for k, v in image_params.items()]
    param_table = Table(param_rows, colWidths=[4 * cm, 8 * cm])
    param_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), simsun),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue)
    ]))

    # 组合图像和参数
    elements.append(Paragraph("病理切片图像", section_style))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(img)
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(param_table)
    elements.append(Spacer(1, 0.5 * cm))

    # 4. 统计表格
    elements.append(Paragraph("汇管区定量分析", section_style))
    stat_table = Table(statistics_data, colWidths=[8 * cm, 8 * cm])
    stat_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), simsun),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT')
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 2 * cm))

    # 5. 签名栏
    signature_style = ParagraphStyle(
        name="Signature",
        fontName="simsun",  # 关键：使用注册的字体名称
        fontSize=12,
        leading=14,
        spaceAfter=20
    )

    signature_text = [
        Paragraph("<font name='simsun'>报告日期：__________________________</font>", signature_style),
        Spacer(1, 0.2 * cm),
        Paragraph("<font name='simsun'>医师签名：__________________________</font>", signature_style)
    ]
    elements.extend(signature_text)

    # 生成PDF
    doc.build(elements)


# ====================
# 执行生成
# ====================
if __name__ == "__main__":
    report_info = {
        "patient_name": '',
        "patient_age": '',
        "patient_gender": '',
        "patient_id": '',
        "image_name": '12',
        "image_resolution": (3880, 6140),
        "portal_num": '',
        "normal_portal_num": '',
        "fibre_portal_num": '',
        "portal_area": '',
        "normal_portal_area": '',
        "fibre_portal_area": ''
    }
    create_report(report_info)
    print(f"PDF报告已生成：{PDF_PATH}")