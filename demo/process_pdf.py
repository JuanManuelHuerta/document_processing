"""
PDF Form Extraction Runner
Handles PDF to image conversion and processes multi-page documents
"""

import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import json
import logging
from typing import List, Dict
from form_extraction_pipeline import (
    FormExtractionPipeline,
    FormType,
    DocumentPreprocessor,
    BarcodeDetector,
    FormClassifier,
    FieldExtractor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFFormProcessor:
    """Processes PDF forms by converting to images and extracting data"""
    
    def __init__(self, output_dir: str = "output", dpi: int = 300):
        """
        Initialize PDF processor
        
        Args:
            output_dir: Directory to save intermediate images and results
            dpi: DPI for PDF to image conversion (300+ recommended)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
        # Initialize extraction pipeline
        barcode_mapping = {
            'FORM-A-': FormType.TYPE_A,
            'FORM-B-': FormType.TYPE_B,
            'FORM-C-': FormType.TYPE_C,
            'MED-': FormType.TYPE_A,
            'INS-': FormType.TYPE_B,
            'REG-': FormType.TYPE_C,
        }
        
        self.pipeline = FormExtractionPipeline(
            use_layoutlm=False,
            barcode_mapping=barcode_mapping
        )
        
        logger.info(f"Initialized PDF processor (DPI: {dpi}, Output: {output_dir})")
    
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF to images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of paths to converted images
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='jpeg',
                grayscale=False
            )
            
            # Save images
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for i, image in enumerate(images, 1):
                image_path = self.output_dir / f"{pdf_name}_page_{i:03d}.jpg"
                image.save(image_path, 'JPEG', quality=95)
                image_paths.append(str(image_path))
                logger.info(f"  Saved page {i}: {image_path}")
            
            logger.info(f"Converted {len(images)} pages")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            raise
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process complete PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with results for all pages
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING PDF: {pdf_path}")
        logger.info(f"{'='*60}\n")
        
        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)
        
        # Process each page
        results = {
            'pdf_path': pdf_path,
            'total_pages': len(image_paths),
            'pages': []
        }
        
        for page_num, image_path in enumerate(image_paths, 1):
            logger.info(f"\n--- Processing Page {page_num}/{len(image_paths)} ---")
            
            try:
                # Extract data from page
                page_result = self.pipeline.process_document(image_path)
                page_result['page_number'] = page_num
                results['pages'].append(page_result)
                
                # Log summary
                logger.info(f"✓ Page {page_num} complete:")
                logger.info(f"  Form Type: {page_result['classification']['form_type']}")
                logger.info(f"  Fields Extracted: {len(page_result['fields'])}")
                if page_result['barcode']['detected']:
                    logger.info(f"  Barcode: {page_result['barcode']['data']}")
                
            except Exception as e:
                logger.error(f"✗ Error processing page {page_num}: {e}")
                results['pages'].append({
                    'page_number': page_num,
                    'error': str(e)
                })
        
        return results
    
    def save_results(self, results: Dict, output_file: str = None):
        """Save extraction results to JSON"""
        if output_file is None:
            pdf_name = Path(results['pdf_path']).stem
            output_file = self.output_dir / f"{pdf_name}_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")
        return output_file
    
    def print_summary(self, results: Dict):
        """Print human-readable summary of extraction results"""
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY")
        print("="*70)
        
        print(f"\nPDF: {results['pdf_path']}")
        print(f"Total Pages: {results['total_pages']}")
        
        for page in results['pages']:
            if 'error' in page:
                print(f"\n❌ Page {page['page_number']}: ERROR - {page['error']}")
                continue
            
            print(f"\n📄 Page {page['page_number']}")
            print(f"   Form Type: {page['classification']['form_type']}")
            print(f"   Confidence: {page['classification']['confidence']:.1%}")
            print(f"   Classification Method: {page['classification']['method']}")
            
            if page['barcode']['detected']:
                print(f"   🔖 Barcode: {page['barcode']['data']} ({page['barcode']['type']})")
            
            if page['fields']:
                print(f"   📝 Fields ({len(page['fields'])}):")
                for field in page['fields']:
                    conf_emoji = "✓" if field['confidence'] > 0.8 else "⚠" if field['confidence'] > 0.5 else "✗"
                    print(f"      {conf_emoji} {field['name']:20} = {field['value']:30} [{field['confidence']:.0%}]")
            else:
                print(f"   ⚠ No fields extracted")
        
        print("\n" + "="*70)


def create_sample_form_image():
    """Create a sample form image for testing if no PDF available"""
    logger.info("Creating sample form image for testing...")
    
    # Create blank form
    width, height = 850, 1100
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(img, "SAMPLE REGISTRATION FORM", (150, 80),
                cv2.FONT_HERSHEY_BOLD, 1, (0, 0, 0), 2)
    
    # Add form fields with labels
    fields = [
        ("Full Name:", 150, 200),
        ("Date of Birth:", 150, 280),
        ("Email:", 150, 360),
        ("Phone:", 150, 440),
        ("Address:", 150, 520),
    ]
    
    for label, x, y in fields:
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.rectangle(img, (x + 200, y - 30), (x + 600, y + 10), (0, 0, 0), 2)
    
    # Add checkbox
    cv2.rectangle(img, (150, 600), (180, 630), (0, 0, 0), 2)
    cv2.putText(img, "I agree to terms and conditions", (200, 625),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add barcode placeholder
    cv2.rectangle(img, (150, 900), (400, 950), (0, 0, 0), 2)
    cv2.putText(img, "FORM-A-12345", (170, 930),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save image
    output_path = Path("output/sample_form.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), img)
    
    logger.info(f"✓ Sample form created: {output_path}")
    return str(output_path)


def main():
    """Main execution function"""
    
    # Initialize processor
    processor = PDFFormProcessor(output_dir="output", dpi=300)
    
    # Check for PDF file
    pdf_path = "data/0086.pdf"
    
    if not Path(pdf_path).exists():
        logger.warning(f"PDF not found: {pdf_path}")
        logger.info("Testing with sample form image instead...")
        
        # Create and process sample form
        sample_image = create_sample_form_image()
        
        # Process single image
        result = processor.pipeline.process_document(sample_image)
        
        # Create results structure
        results = {
            'pdf_path': sample_image,
            'total_pages': 1,
            'pages': [{**result, 'page_number': 1}]
        }
        
    else:
        # Process actual PDF
        results = processor.process_pdf(pdf_path)
    
    # Save and display results
    output_file = processor.save_results(results)
    processor.print_summary(results)
    
    return results, output_file


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PDF FORM EXTRACTION SYSTEM")
    print("="*70)
    
    try:
        results, output_file = main()
        print(f"\n✅ Processing complete! Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
