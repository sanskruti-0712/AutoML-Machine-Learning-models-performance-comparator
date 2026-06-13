import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(dataset_summary: dict, metrics_summary: dict, 
                        best_model: str, problem_type: str, 
                        preprocessing_steps: list, advisor_text: str) -> io.BytesIO:
    """
    Generates a beautifully formatted PDF report summarizing the AutoML run.
    Returns the PDF content as a BytesIO stream.
    """
    buffer = io.BytesIO()
    
    # 1. Setup Document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    # 2. Setup Styles
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#312E81")   # Deep Indigo
    secondary_color = colors.HexColor("#4F46E5") # Indigo
    text_color = colors.HexColor("#1F2937")      # Dark Grey
    bg_light = colors.HexColor("#F9FAFB")        # Off-white
    accent_color = colors.HexColor("#D97706")    # Amber
    
    # Safely create styles
    def get_style(name, parent, **kwargs):
        if name in styles:
            return styles[name]
        style = ParagraphStyle(name, parent=parent, **kwargs)
        styles.add(style)
        return style
        
    title_style = get_style(
        'ReportTitle', 
        styles['Heading1'], 
        fontSize=24, 
        leading=28, 
        textColor=primary_color, 
        spaceAfter=15
    )
    
    h1_style = get_style(
        'ReportH1', 
        styles['Heading2'], 
        fontSize=16, 
        leading=20, 
        textColor=primary_color, 
        spaceBefore=15, 
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = get_style(
        'ReportBody', 
        styles['BodyText'], 
        fontSize=10, 
        leading=14, 
        textColor=text_color, 
        spaceAfter=6
    )
    
    body_bold = get_style(
        'ReportBodyBold', 
        body_style, 
        fontName='Helvetica-Bold'
    )
    
    list_style = get_style(
        'ReportList', 
        body_style, 
        leftIndent=15, 
        bulletIndent=5, 
        spaceAfter=4
    )
    
    callout_style = get_style(
        'ReportCallout', 
        body_style, 
        fontSize=11, 
        leading=15, 
        textColor=primary_color, 
        spaceBefore=10, 
        spaceAfter=10
    )

    story = []
    
    # --- HEADER SECTION ---
    story.append(Paragraph("AutoML Platform Evaluation Report", title_style))
    story.append(Paragraph("A comprehensive performance comparison of machine learning models.", body_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # --- DATASET SUMMARY ---
    story.append(Paragraph("1. Dataset Summary", h1_style))
    
    data_summary_table_data = [
        [Paragraph("Metadata Attribute", body_bold), Paragraph("Value", body_bold)],
        [Paragraph("Target Column", body_style), Paragraph(str(dataset_summary.get("detected_target", "N/A")), body_style)],
        [Paragraph("Problem Type", body_style), Paragraph(str(problem_type).capitalize(), body_style)],
        [Paragraph("Total Dataset Rows", body_style), Paragraph(f"{dataset_summary.get('total_rows', 0):,}", body_style)],
        [Paragraph("Total Dataset Columns", body_style), Paragraph(str(dataset_summary.get("total_cols", 0)), body_style)],
        [Paragraph("Target Cardinality", body_style), Paragraph(str(dataset_summary.get("unique_target_values", "N/A")), body_style)]
    ]
    
    t_summary = Table(data_summary_table_data, colWidths=[200, 300])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
    ]))
    
    # Wrap headers textcolor
    for i in range(2):
        data_summary_table_data[0][i].style.textColor = colors.white
        
    story.append(t_summary)
    story.append(Spacer(1, 0.2 * inch))
    
    # --- PREPROCESSING STEPS ---
    story.append(Paragraph("2. Smart Preprocessing Pipeline", h1_style))
    story.append(Paragraph("The platform automatically executed the following pre-processing stages to prepare the data:", body_style))
    
    if preprocessing_steps:
        for idx, step in enumerate(preprocessing_steps):
            step_text = f"<b>{step.get('step')}:</b> {step.get('details')}"
            story.append(Paragraph(f"&bull; {step_text}", list_style))
    else:
        story.append(Paragraph("No explicit preprocessing log recorded.", list_style))
        
    story.append(Spacer(1, 0.2 * inch))
    
    # --- MODEL PERFORMANCE COMPARISON ---
    story.append(Paragraph("3. Model Performance Comparison", h1_style))
    
    is_classification = problem_type == "classification"
    
    if is_classification:
        header_row = [
            Paragraph("Model Name", body_bold),
            Paragraph("Accuracy", body_bold),
            Paragraph("F1-Score", body_bold),
            Paragraph("Precision", body_bold),
            Paragraph("Recall", body_bold),
            Paragraph("Train Time (s)", body_bold)
        ]
    else:
        header_row = [
            Paragraph("Model Name", body_bold),
            Paragraph("R2 Score", body_bold),
            Paragraph("MAE", body_bold),
            Paragraph("RMSE", body_bold),
            Paragraph("Train Time (s)", body_bold)
        ]
        
    table_data = [header_row]
    
    for name, res in metrics_summary.items():
        metrics = res.get("metrics", {})
        if is_classification:
            row = [
                Paragraph(f"<b>{name}</b>" if name == best_model else name, body_style),
                Paragraph(f"{metrics.get('accuracy', 0)*100:.2f}%", body_style),
                Paragraph(f"{metrics.get('f1', 0)*100:.2f}%", body_style),
                Paragraph(f"{metrics.get('precision', 0)*100:.2f}%", body_style),
                Paragraph(f"{metrics.get('recall', 0)*100:.2f}%", body_style),
                Paragraph(f"{metrics.get('train_time_sec', 0):.3f}s", body_style)
            ]
        else:
            row = [
                Paragraph(f"<b>{name}</b>" if name == best_model else name, body_style),
                Paragraph(f"{metrics.get('r2', 0):.4f}", body_style),
                Paragraph(f"{metrics.get('mae', 0):.4f}", body_style),
                Paragraph(f"{metrics.get('rmse', 0):.4f}", body_style),
                Paragraph(f"{metrics.get('train_time_sec', 0):.3f}s", body_style)
            ]
        table_data.append(row)
        
    # Set text color in header row elements to white
    for item in header_row:
        item.style.textColor = colors.white
        
    col_widths = [120, 76, 76, 76, 76, 76] if is_classification else [150, 87, 87, 87, 87]
    t_perf = Table(table_data, colWidths=col_widths)
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
    ]))
    
    story.append(t_perf)
    
    # Highlight Best Model
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"<b>Best Performing Model:</b> {best_model}", callout_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # --- AI ADVISOR DETAILS ---
    story.append(Paragraph("4. AI Data Science Advisor Insights", h1_style))
    
    # Clean advisor text from markdown headings to represent cleanly in ReportLab
    lines = advisor_text.split('\n')
    cleaned_paragraphs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("###") or line.startswith("####"):
            header_txt = line.replace("###", "").replace("####", "").strip()
            cleaned_paragraphs.append(Paragraph(f"<b>{header_txt}</b>", get_style('AdvisorSubH', h1_style, fontSize=11, spaceBefore=8, spaceAfter=4)))
        elif line.startswith("-") or line.startswith("*"):
            bullet_txt = line[1:].strip()
            cleaned_paragraphs.append(Paragraph(f"&bull; {bullet_txt}", list_style))
        elif line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            list_txt = line[2:].strip()
            cleaned_paragraphs.append(Paragraph(f"{line[:2]} {list_txt}", list_style))
        else:
            cleaned_paragraphs.append(Paragraph(line, body_style))
            
    # Wrap advisor section together to prevent page breaks mid-section
    story.append(KeepTogether(cleaned_paragraphs))
    
    # Build Document
    doc.build(story)
    
    buffer.seek(0)
    return buffer
