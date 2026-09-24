from pathlib import Path
import re, html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
root=Path('/Users/novaz/Desktop/qml2.0')
font='/Users/novaz/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/fonts/DejaVuSans.ttf'
pdfmetrics.registerFont(TTFont('DV',font))
text=(root/'output/code_audit_20260924/CODE_AUDIT.md').read_text()
styles={
'p':ParagraphStyle('p',fontName='DV',fontSize=9,leading=13,spaceAfter=8),
'h':ParagraphStyle('h',fontName='DV',fontSize=15,leading=20,spaceBefore=12,spaceAfter=10,textColor=colors.HexColor('#163c54')),
'title':ParagraphStyle('title',fontName='DV',fontSize=23,leading=29,spaceAfter=14,textColor=colors.HexColor('#163c54')),
'cell':ParagraphStyle('cell',fontName='DV',fontSize=8.5,leading=12),
'head':ParagraphStyle('head',fontName='DV',fontSize=8.5,leading=12,textColor=colors.white)}
def clean(s):
 s=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',s)
 s=s.replace('—','-').replace('–','-').replace('−','-').replace('**','').replace('`','')
 return html.escape(s)
def p(s,style='p'):return Paragraph(clean(s),styles[style])
story=[]; lines=text.splitlines(); i=0
while i<len(lines):
 l=lines[i]
 if l.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].startswith('|'):
   row=[v.strip() for v in lines[i].strip('|').split('|')]
   if not all(re.fullmatch(r'[: -]+',v) for v in row):rows.append(row)
   i+=1
  widths=[90,167,513] if rows[0][0]=='Layer' else [238,342,190]
  data=[[p(v,'head' if j==0 else 'cell') for v in row] for j,row in enumerate(rows)]
  table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#163c54')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#163c54')),('INNERGRID',(0,1),(-1,-1),.25,colors.HexColor('#d4dfe5'))]))
  story.extend([table,Spacer(1,12)]);continue
 if l.startswith('# '):story.append(p(l[2:],'title'))
 elif l.startswith('## '):
  if 'Computational-stage' in l or 'Numerical verification' in l:story.append(PageBreak())
  story.append(p(l[3:],'h'))
 elif l.strip():story.append(p(l))
 i+=1
out=root/'output/pdf/qml2_codes_audit_table.pdf'
def footer(c,d):
 c.setFont('DV',8);c.setFillColor(colors.HexColor('#526979'))
 c.drawString(36,20,'qml2.0/codes | Code-only audit | 24 September 2026')
 c.drawRightString(806,20,str(d.page))
doc=SimpleDocTemplate(str(out),pagesize=landscape(A4),rightMargin=35,leftMargin=36,topMargin=30,bottomMargin=35,title='qml2.0/codes - Computational Pipeline Audit',author='Codex')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
import fitz
pdf=fitz.open(out)
alltext='\n'.join(page.get_text() for page in pdf)
assert all(str(n)+' ' in alltext for n in range(1,38))
for j,page in enumerate(pdf):
 page.get_pixmap(matrix=fitz.Matrix(1,1)).save(root/f'tmp/pdfs/page-{j+1:02}.png')
print(f'{out}\nPages: {len(pdf)}; all 37 stage labels present')
from PIL import Image, ImageOps, ImageDraw
thumbs=[]
for path in sorted((root/'tmp/pdfs').glob('page-*.png')):
 im=Image.open(path).convert('RGB');im.thumbnail((600,425));thumbs.append(im)
contact=Image.new('RGB',(1200,450*((len(thumbs)+1)//2)), '#cbd5dd')
for k,im in enumerate(thumbs):contact.paste(im,((k%2)*600,(k//2)*450))
contact.save(root/'tmp/pdfs/review.png')
