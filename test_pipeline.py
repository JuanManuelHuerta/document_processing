"""
Test script for Form Extraction Pipeline
Demonstrates usage and generates sample output
"""

import json
import logging
from pathlib import Path
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


def test_preprocessing():
    """Test Stage 1: Preprocessing"""
    print("\n" + "="*60)
    print("TEST 1: PREPROCESSING")
    print("="*60)
    
    preprocessor = DocumentPreprocessor()
    
    # Test with sample image path (replace with actual path)
    test_image = "test_data/sample_form.jpg"
    
    if Path(test_image).exists():
        processed = preprocessor.process(test_image)
        print(f"✓ Successfully preprocessed image")
        print(f"  Output shape: {processed.shape}")
        print(f"  Data type: {processed.dtype}")
    else:
        print(f"⚠ Test image not found: {test_image}")
        print(f"  Create test_data directory and add sample images to test")


def test_barcode_detection():
    """Test Stage 2: Barcode Detection"""
    print("\n" + "="*60)
    print("TEST 2: BARCODE DETECTION")
    print("="*60)
    
    detector = BarcodeDetector()
    
    # Mock test data
    print("Supported barcode types:")
    for bc_type in detector.supported_types:
        print(f"  - {bc_type}")
    
    print("\n✓ Barcode detector initialized")
    print("  Note: Add actual test images with barcodes to test detection")


def test_classification():
    """Test Stage 3: Form Classification"""
    print("\n" + "="*60)
    print("TEST 3: FORM CLASSIFICATION")
    print("="*60)
    
    # Define barcode mapping
    barcode_mapping = {
        'MED-': FormType.TYPE_A,
        'INS-': FormType.TYPE_B,
        'REG-': FormType.TYPE_C,
    }
    
    classifier = FormClassifier(barcode_mapping=barcode_mapping)
    
    print("Barcode mapping configured:")
    for prefix, form_type in barcode_mapping.items():
        print(f"  {prefix}* → {form_type.value}")
    
    # Test barcode classification logic
    from form_extraction_pipeline import BarcodeResult
    
    test_cases = [
        BarcodeResult(data='MED-12345', type='CODE128', location=(0, 0, 100, 50)),
        BarcodeResult(data='INS-98765', type='QRCODE', location=(0, 0, 100, 100)),
        BarcodeResult(data='UNKNOWN-123', type='CODE39', location=(0, 0, 100, 50)),
    ]
    
    print("\nTesting barcode classification:")
    for barcode in test_cases:
        form_type = classifier._classify_by_barcode(barcode)
        print(f"  {barcode.data} → {form_type.value}")


def test_field_extraction():
    """Test Stage 4: Field Extraction"""
    print("\n" + "="*60)
    print("TEST 4: FIELD EXTRACTION")
    print("="*60)
    
    extractor = FieldExtractor(use_layoutlm=False)
    
    print("Template-based extraction configured")
    print(f"Available form templates: {len(extractor.form_templates)}")
    
    for form_type, template in extractor.form_templates.items():
        print(f"\n  {form_type.value}:")
        print(f"    Fields: {len(template)}")
        for field in template:
            print(f"      - {field.name} ({field.field_type})")


def test_full_pipeline():
    """Test complete pipeline integration"""
    print("\n" + "="*60)
    print("TEST 5: FULL PIPELINE")
    print("="*60)
    
    # Configure pipeline
    barcode_mapping = {
        'FORM-A-': FormType.TYPE_A,
        'FORM-B-': FormType.TYPE_B,
        'FORM-C-': FormType.TYPE_C,
    }
    
    pipeline = FormExtractionPipeline(
        use_layoutlm=False,
        barcode_mapping=barcode_mapping
    )
    
    print("Pipeline initialized with components:")
    print("  ✓ Preprocessor")
    print("  ✓ Barcode Detector")
    print("  ✓ Form Classifier")
    print("  ✓ Field Extractor")
    
    # Test with sample image if available
    test_images = [
        "test_data/sample_medical_form.jpg",
        "test_data/sample_insurance_form.jpg",
        "test_data/sample_registration_form.jpg",
    ]
    
    results = []
    for test_image in test_images:
        if Path(test_image).exists():
            print(f"\nProcessing: {test_image}")
            try:
                result = pipeline.process_document(test_image)
                results.append(result)
                print(f"  ✓ Success")
                print(f"    Form Type: {result['classification']['form_type']}")
                print(f"    Fields Extracted: {len(result['fields'])}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"\n⚠ Test image not found: {test_image}")
    
    if not results:
        print("\nNo test images found. Testing with mock result...")
        mock_result = generate_mock_result()
        results.append(mock_result)
        print_extraction_result(mock_result)
    
    # Save results
    if results:
        output_file = "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")


def generate_mock_result():
    """Generate a mock extraction result for demonstration"""
    return {
        'image_path': 'test_data/sample_form.jpg',
        'barcode': {
            'detected': True,
            'data': 'FORM-A-12345',
            'type': 'CODE128',
        },
        'classification': {
            'form_type': 'form_type_a',
            'confidence': 1.0,
            'method': 'barcode',
        },
        'fields': [
            {
                'name': 'full_name',
                'value': 'John Doe',
                'confidence': 0.92,
            },
            {
                'name': 'date_of_birth',
                'value': '01/15/1985',
                'confidence': 0.88,
            },
            {
                'name': 'agree_terms',
                'value': 'checked',
                'confidence': 0.95,
            },
        ]
    }


def print_extraction_result(result):
    """Pretty print extraction results"""
    print("\n" + "─"*60)
    print("EXTRACTION RESULT")
    print("─"*60)
    
    print(f"\nDocument: {result['image_path']}")
    
    print("\n📊 Classification:")
    print(f"  Form Type: {result['classification']['form_type']}")
    print(f"  Confidence: {result['classification']['confidence']:.2%}")
    print(f"  Method: {result['classification']['method']}")
    
    if result['barcode']['detected']:
        print("\n🔖 Barcode:")
        print(f"  Type: {result['barcode']['type']}")
        print(f"  Data: {result['barcode']['data']}")
    
    print(f"\n📝 Extracted Fields ({len(result['fields'])}):")
    for field in result['fields']:
        confidence_bar = "█" * int(field['confidence'] * 10) + "░" * (10 - int(field['confidence'] * 10))
        print(f"  {field['name']:20} = {field['value']:20} [{confidence_bar}] {field['confidence']:.2%}")
    
    print("\n" + "─"*60)


def benchmark_pipeline():
    """Benchmark pipeline performance"""
    print("\n" + "="*60)
    print("BENCHMARK: PIPELINE PERFORMANCE")
    print("="*60)
    
    import time
    
    # Simulate processing times
    stages = {
        'Preprocessing': (0.5, 1.0),
        'Barcode Detection': (0.1, 0.3),
        'Form Classification': (0.2, 0.5),
        'Field Extraction (Template)': (2.0, 4.0),
        'Field Extraction (LayoutLM)': (3.0, 6.0),
    }
    
    print("\nExpected processing times per page:")
    print(f"{'Stage':<30} {'Min':>8} {'Max':>8} {'Avg':>8}")
    print("─" * 60)
    
    total_min = 0
    total_max = 0
    
    for stage, (min_time, max_time) in stages.items():
        avg_time = (min_time + max_time) / 2
        print(f"{stage:<30} {min_time:>7.2f}s {max_time:>7.2f}s {avg_time:>7.2f}s")
        
        if 'LayoutLM' not in stage:
            total_min += min_time
            total_max += max_time
    
    print("─" * 60)
    total_avg = (total_min + total_max) / 2
    print(f"{'Total (Template-based)':<30} {total_min:>7.2f}s {total_max:>7.2f}s {total_avg:>7.2f}s")
    
    print("\n📈 Throughput estimates:")
    print(f"  Sequential processing: {3600/total_avg:.0f} pages/hour")
    print(f"  4 parallel workers: {4*3600/total_avg:.0f} pages/hour")
    print(f"  8 parallel workers: {8*3600/total_avg:.0f} pages/hour")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FORM EXTRACTION PIPELINE - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Barcode Detection", test_barcode_detection),
        ("Form Classification", test_classification),
        ("Field Extraction", test_field_extraction),
        ("Full Pipeline", test_full_pipeline),
        ("Benchmarks", benchmark_pipeline),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print("\nNext Steps:")
    print("1. Add test images to test_data/ directory")
    print("2. Configure barcode mapping in config.yaml")
    print("3. Define field templates for your form types")
    print("4. Fine-tune LayoutLM model on your labeled data")
    print("5. Implement validation rules (Stage 5)")
    print("\n")


if __name__ == "__main__":
    main()
