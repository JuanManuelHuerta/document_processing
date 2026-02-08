# Setup Guide for Processing 0086.pdf

## Quick Start

### 1. Directory Structure
```
your-project/
├── data/
│   └── 0086.pdf              # Place your PDF here
├── output/                   # Results will be saved here
├── form_extraction_pipeline.py
├── process_pdf.py
├── demo_pipeline.py
├── config.yaml
├── requirements.txt
└── README.md
```

### 2. System Requirements

**Required System Packages:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libzbar0

# macOS
brew install tesseract poppler zbar

# Windows (using Chocolatey)
choco install tesseract poppler
```

**Python Dependencies:**
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Processing Your PDF (data/0086.pdf)

**Option A: Run Demo (No Dependencies Required)**
```bash
python3 demo_pipeline.py
```
This shows how the pipeline works with simulated data.

**Option B: Run Full Pipeline (Requires Dependencies)**
```bash
# 1. Place your PDF
mkdir -p data
cp /path/to/your/0086.pdf data/0086.pdf

# 2. Run processing
python3 process_pdf.py

# 3. View results
cat output/0086_extraction_results.json
```

### 4. Configuration for Your Forms

Edit `config.yaml` to match your form types:

```yaml
barcode_mapping:
  '0086-': 'medical_form'  # Map barcode prefix to form type
  'MED-': 'medical_form'
  'INS-': 'insurance_claim'

form_types:
  medical_form:
    name: "Medical Form 0086"
    fields:
      - name: "patient_name"
        x: 150
        y: 220
        width: 400
        height: 50
        type: "text"
        required: true
      # Add more fields...
```

## Processing Pipeline Flow

```
PDF Document (0086.pdf)
    ↓
[Convert to Images]  (300 DPI JPEG)
    ↓
For each page:
    ↓
[Stage 1: Preprocess]
  • Deskew
  • Enhance contrast
  • Denoise
  • Binarize
    ↓
[Stage 2: Detect Barcode/QR]
  • pyzbar detection
  • Extract form identifier
    ↓
[Stage 3: Classify Form]
  • Barcode mapping (primary)
  • CNN classification (fallback)
    ↓
[Stage 4: Extract Fields]
  • Template-based OR
  • LayoutLM-based
  • OCR handwriting
    ↓
[Results JSON]
{
  "page": 1,
  "form_type": "medical_form",
  "fields": [...]
}
```

## Expected Output Structure

```json
{
  "pdf_path": "data/0086.pdf",
  "total_pages": 2,
  "pages": [
    {
      "page_number": 1,
      "barcode": {
        "detected": true,
        "data": "0086-001",
        "type": "CODE128"
      },
      "classification": {
        "form_type": "medical_form",
        "confidence": 1.0,
        "method": "barcode"
      },
      "fields": [
        {
          "name": "patient_name",
          "value": "John Doe",
          "confidence": 0.92
        },
        {
          "name": "date_of_birth",
          "value": "01/15/1985",
          "confidence": 0.88
        }
        // ... more fields
      ]
    }
    // ... more pages
  ]
}
```

## Improving Accuracy

### For Form 0086 Specifically:

1. **Measure Field Coordinates**
   - Open 0086.pdf in an image editor
   - Note exact pixel positions of each field
   - Update `config.yaml` with these coordinates

2. **Create Field Template**
   ```python
   templates = {
       'medical_form_0086': [
           FieldBox("patient_id", x=100, y=150, width=200, height=40, field_type="text"),
           FieldBox("patient_name", x=100, y=200, width=400, height=50, field_type="text"),
           # ... measure all fields
       ]
   }
   ```

3. **Optimize OCR Settings**
   - For typed text: Use `--psm 7` (single line)
   - For handwriting: Use cloud OCR (Google Vision or AWS Textract)
   - For checkboxes: Adjust pixel density threshold

4. **Fine-tune LayoutLM (Recommended)**
   ```bash
   # Annotate 50-100 examples of form 0086
   # Train LayoutLMv3 on your specific form
   # Replace template-based extraction
   ```

## Troubleshooting

### Issue: Low OCR Accuracy (<70%)
**Solutions:**
- Increase scan DPI (use 600 DPI instead of 300)
- Check preprocessing - may be over-aggressive
- Switch to Google Cloud Vision for handwriting
- Verify form isn't skewed after preprocessing

### Issue: Barcode Not Detected
**Solutions:**
- Check barcode is clear and not damaged
- Try different contrast/brightness settings
- Manually verify with: `zbarimg image.jpg`
- Consider QR code instead of barcode

### Issue: Wrong Form Classification
**Solutions:**
- Verify barcode mapping in config.yaml
- Add more training examples for CNN classifier
- Check for similar-looking forms
- Use barcode on every page

### Issue: Fields Extracted from Wrong Location
**Solutions:**
- Re-measure field coordinates precisely
- Check if form has multiple versions/layouts
- Verify deskewing isn't shifting content
- Use anchor points to auto-calibrate

## Production Deployment

### Scaling for High Volume

```python
# Use async processing
from concurrent.futures import ProcessPoolExecutor

def process_batch(pdf_paths):
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_pdf, pdf_paths))
    return results
```

### API Service

```python
from fastapi import FastAPI, UploadFile, BackgroundTasks
import uvicorn

app = FastAPI()

@app.post("/extract/0086")
async def extract_0086(file: UploadFile, background_tasks: BackgroundTasks):
    # Save file
    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    
    # Process in background
    background_tasks.add_task(process_pdf, pdf_path)
    
    return {"status": "processing", "job_id": "..."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Monitoring

```python
# Add logging and metrics
import logging
from prometheus_client import Counter, Histogram

extraction_counter = Counter('forms_extracted', 'Total forms processed')
extraction_time = Histogram('extraction_seconds', 'Time to extract form')

@extraction_time.time()
def process_form(pdf_path):
    result = pipeline.process_document(pdf_path)
    extraction_counter.inc()
    return result
```

## Next Steps

1. ✅ Place your PDF at `data/0086.pdf`
2. ✅ Run demo: `python3 demo_pipeline.py`
3. ✅ Install dependencies
4. ✅ Run full pipeline: `python3 process_pdf.py`
5. ⬜ Measure field coordinates for form 0086
6. ⬜ Update config.yaml with exact coordinates
7. ⬜ Test with multiple sample forms
8. ⬜ Fine-tune LayoutLM on labeled examples
9. ⬜ Add validation rules (Stage 5)
10. ⬜ Deploy to production

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review troubleshooting section above
3. Examine the demo output to understand expected behavior
4. Test with simpler forms first to validate setup

## Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Processing Time | 3-8 seconds per page |
| OCR Accuracy (Printed) | 95-99% |
| OCR Accuracy (Handwritten) | 70-85% (Tesseract) |
| OCR Accuracy (Handwritten) | 85-95% (Cloud services) |
| Barcode Detection | 99%+ |
| Form Classification | 90-98% |

Good luck with your form extraction! 🚀
