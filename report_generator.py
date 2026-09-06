"""PDF report generation for Crime Pattern Analysis."""
import io
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

def _png(fig):
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=150,bbox_inches="tight"); plt.close(fig); b.seek(0); return b

def _chart_series(data):
    """Convert either a Series or the dashboard's DataFrame results into a label/value Series."""
    if data is None or len(data) == 0:
        return None
    if isinstance(data, pd.Series):
        s = data.copy()
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return None
        value_col = "count" if "count" in data.columns else "crime_count" if "crime_count" in data.columns else None
        if value_col is None:
            numeric = data.select_dtypes(include="number").columns
            if len(numeric) == 0:
                return None
            value_col = numeric[-1]
        label_cols = [c for c in data.columns if c != value_col]
        if not label_cols:
            return None
        label_col = label_cols[0]
        out = data[[label_col, value_col]].copy()
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        out = out.dropna(subset=[value_col])
        return pd.Series(out[value_col].values, index=out[label_col].astype(str).values)
    return None

def _bar(s,title,xlabel):
    s = _chart_series(s)
    if s is None or len(s)==0: return None
    s = s.head(10).sort_values(ascending=True)
    fig,ax=plt.subplots(figsize=(8,4.2)); ax.barh(s.index.astype(str),s.values); ax.set_title(title,fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel("Number of Records"); ax.grid(axis="x",alpha=.2); fig.tight_layout(); return _png(fig)

def _line(s,title,xlabel):
    s = _chart_series(s)
    if s is None or len(s)==0: return None
    fig,ax=plt.subplots(figsize=(8,4.2)); ax.plot(s.index.astype(str),s.values,marker="o",linewidth=2); ax.set_title(title,fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel("Number of Records"); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout(); return _png(fig)

def _df_table(df,max_rows=10):
    if df is None or df.empty: return None
    v=df.head(max_rows).copy(); v=v.astype(str).map(lambda x:x[:40]); data=[list(v.columns)]+v.values.tolist(); t=Table(data,repeatRows=1,hAlign="LEFT"); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dc2626")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#fecaca")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fff5f5")]),("VALIGN",(0,0),(-1,-1),"TOP")])); return t

def generate_pdf_report(raw_info,filtered_df,top_crimes,top_locations,by_year,by_month,by_weekday,by_hour,hotspots,risk_df,filter_summary=None):
    b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36,title="Crime Pattern Analysis Report")
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name="RT",parent=styles["Title"],alignment=TA_CENTER,textColor=colors.HexColor("#991b1b"),fontSize=20)); styles.add(ParagraphStyle(name="RS",parent=styles["Heading2"],textColor=colors.HexColor("#991b1b"),fontSize=14,spaceBefore=10,spaceAfter=7)); styles.add(ParagraphStyle(name="SM",parent=styles["BodyText"],fontSize=8.5,leading=12))
    st=[]; st.append(Paragraph("Crime Pattern Analysis Report",styles["RT"])); st.append(Paragraph("Using Big Data Analytics",styles["Heading3"])); st.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",styles["SM"])); st.append(Spacer(1,10)); st.append(Paragraph("This report summarizes descriptive crime-pattern analysis performed on the dataset currently loaded and filtered in the dashboard. Results describe historical aggregate patterns and should not be interpreted as individual risk scores or predictive-policing decisions.",styles["BodyText"]))
    st.append(Paragraph("1. Dataset and Preprocessing Summary",styles["RS"])); raw=raw_info.get("raw_shape",("N/A","N/A")); clean=raw_info.get("clean_shape",("N/A","N/A")); rows=[["Measure","Result"],["Raw dataset rows",str(raw[0])],["Raw dataset columns",str(raw[1])],["Clean dataset rows",str(clean[0])],["Clean dataset columns",str(clean[1])],["Rows in current analysis",str(len(filtered_df))],["Columns used",", ".join(map(str,filtered_df.columns[:12]))+(' ...' if len(filtered_df.columns)>12 else '')]]; t=Table(rows,colWidths=[2.2*inch,4.6*inch]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dc2626")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#fecaca")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fff5f5")]),("FONTSIZE",(0,0),(-1,-1),8.5)])); st.append(t)
    if raw_info.get("warnings"): st.append(Paragraph("Preprocessing notes: "+" ".join(raw_info["warnings"][:6]),styles["SM"]))
    if filter_summary: st.append(Paragraph("Applied Filters",styles["Heading3"])); ft=Table([["Filter","Selection"]]+[[str(k),str(v)] for k,v in filter_summary.items()],colWidths=[2.2*inch,4.6*inch]); ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#991b1b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#fecaca")),("FONTSIZE",(0,0),(-1,-1),8)])); st.append(ft)
    st.append(Paragraph("2. Key Analysis Results",styles["RS"])); mc=filtered_df["crime_type"].mode().iloc[0] if "crime_type" in filtered_df and not filtered_df.empty else "N/A"; ml=filtered_df["city"].mode().iloc[0] if "city" in filtered_df and not filtered_df.empty else "N/A"; key=[["Indicator","Result"],["Total records analyzed",str(len(filtered_df))],["Distinct crime types",str(filtered_df["crime_type"].nunique()) if "crime_type" in filtered_df else "N/A"],["Distinct locations",str(filtered_df["city"].nunique()) if "city" in filtered_df else "N/A"],["Most common crime type",str(mc)],["Highest-frequency location",str(ml)]]; kt=Table(key,colWidths=[2.8*inch,4*inch]); kt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dc2626")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#fecaca")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fff5f5")]),("FONTSIZE",(0,0),(-1,-1),8.5)])); st.append(kt)
    st.append(Paragraph("3. Visual Analysis",styles["RS"])); specs=[(top_crimes,"Crime Type Distribution","Crime type",False),(top_locations,"Top Locations by Crime Count","Location",False),(by_year,"Crime Trend by Year","Year",True),(by_month,"Crime Distribution by Month","Month",False),(by_weekday,"Crime Distribution by Day of Week","Day",False),(by_hour,"Crime Distribution by Hour","Hour",False)]
    for s,title,x,isline in specs:
        img=_line(s,title,x) if isline else _bar(s,title,x)
        if img: st.append(Paragraph(title,styles["Heading3"])); st.append(Image(img,width=6.6*inch,height=3.45*inch)); st.append(Spacer(1,5))
    st.append(PageBreak()); st.append(Paragraph("4. Hotspot Analysis",styles["RS"])); st.append(Paragraph("Hotspots are descriptive high-frequency locations within the selected dataset, not official designations or forecasts.",styles["BodyText"])); ht=_df_table(hotspots); rt=_df_table(risk_df); 
    if ht: st.append(Spacer(1,6)); st.append(ht)
    if rt: st.append(Spacer(1,12)); st.append(Paragraph("Location Risk-Level Buckets",styles["Heading3"])); st.append(rt)
    st.append(Paragraph("5. Interpretation / Findings",styles["RS"])); findings=[]
    crime_s = _chart_series(top_crimes)
    location_s = _chart_series(top_locations)
    year_s = _chart_series(by_year)
    month_s = _chart_series(by_month)
    weekday_s = _chart_series(by_weekday)
    hour_s = _chart_series(by_hour)
    if len(filtered_df): findings.append(f"The analysis contains {len(filtered_df):,} crime records after the selected filters and preprocessing.")
    if crime_s is not None and len(crime_s): findings.append(f"The most frequent crime category is {crime_s.index[0]} with {int(crime_s.iloc[0]):,} records.")
    if location_s is not None and len(location_s): findings.append(f"The highest-frequency location is {location_s.index[0]} with {int(location_s.iloc[0]):,} records.")
    if year_s is not None and len(year_s): findings.append(f"The highest annual crime count occurs in {year_s.idxmax()}.")
    if month_s is not None and len(month_s): findings.append(f"The month with the highest recorded crime count is {month_s.idxmax()}.")
    if weekday_s is not None and len(weekday_s): findings.append(f"The weekday with the highest recorded crime count is {weekday_s.idxmax()}.")
    if hour_s is not None and len(hour_s): findings.append(f"The highest recorded count by hour is at {hour_s.idxmax()}:00.")
    for f in findings: st.append(Paragraph("• "+f,styles["BodyText"])); st.append(Spacer(1,3))
    st.append(Paragraph("6. Conclusion",styles["RS"])); st.append(Paragraph("The dashboard provides an integrated workflow for dataset ingestion, preprocessing, descriptive crime-pattern analysis, temporal analysis, and hotspot exploration. This report captures the results for the exact filtered dataset state at the time of generation.",styles["BodyText"])); st.append(Spacer(1,12)); st.append(Paragraph("Academic-use note: historical aggregate patterns do not establish causation and should not be used to target individuals or communities.",styles["SM"])); doc.build(st); return b.getvalue()
