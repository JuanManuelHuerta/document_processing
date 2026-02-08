"""
Production Pipeline for Multi-Page Handwritten Form Extraction
Stages: Preprocessing -> Barcode Detection -> Form Classification -> Field Extraction
"""

import cv2
import numpy as np
from pyzbar import pyzbar
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from typing import List, Dict, Tuple, Optional
import pytesseract
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STAGE 1: PREPROCESSING
# ============================================================================

class DocumentPreprocessor:
    """Handles image preprocessing: deskewing, dewarping, enhancement"""
    
    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi
    
    def process(self, image_path: str) -> np.ndarray:
        """Main preprocessing pipeline"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance image quality
        enhanced = self._enhance_image(gray)
        
        # Deskew
        deskewed = self._deskew(enhanced)
        
        # Denoise
        denoised = self._denoise(deskewed)
        
        # Binarize for better OCR
        binary = self._adaptive_threshold(denoised)
        
        logger.info(f"Preprocessed image: {image_path}")
        return binary
    
    def _enhance_image(self, gray: np.ndarray) -> np.ndarray:
        """Enhance contrast and brightness"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew angle"""
        # Use Hough transform to detect lines
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            return image
        
        # Calculate dominant angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:  # Filter outliers
                angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        
        # Rotate image if skew is significant
        if abs(median_angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), 
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            logger.info(f"Deskewed by {median_angle:.2f} degrees")
            return rotated
        
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges"""
        return cv2.fastNlMeansDenoising(image, None, h=10, 
                                       templateWindowSize=7, 
                                       searchWindowSize=21)
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization"""
        return cv2.adaptiveThreshold(image, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)


# ============================================================================
# STAGE 2: BARCODE/QR CODE DETECTION
# ============================================================================

@dataclass
class BarcodeResult:
    """Result from barcode/QR detection"""
    data: str
    type: str
    location: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float = 1.0


class BarcodeDetector:
    """Detects and decodes barcodes and QR codes"""
    
    def __init__(self):
        self.supported_types = ['QRCODE', 'CODE128', 'CODE39', 'EAN13', 'EAN8']
    
    def detect(self, image: np.ndarray) -> Optional[BarcodeResult]:
        """Detect and decode barcode/QR code in image"""
        # pyzbar works with PIL images
        pil_image = Image.fromarray(image)
        decoded_objects = pyzbar.decode(pil_image)
        
        if not decoded_objects:
            logger.info("No barcode detected")
            return None
        
        # Take the first detected code (usually forms have one identifier)
        obj = decoded_objects[0]
        
        result = BarcodeResult(
            data=obj.data.decode('utf-8'),
            type=obj.type,
            location=(obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height)
        )
        
        logger.info(f"Detected {result.type}: {result.data}")
        return result


# ============================================================================
# STAGE 3: FORM CLASSIFICATION
# ============================================================================

class FormType(Enum):
    """Enum for different form types - customize based on your forms"""
    TYPE_A = "form_type_a"
    TYPE_B = "form_type_b"
    TYPE_C = "form_type_c"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result from form classification"""
    form_type: FormType
    confidence: float
    method: str  # 'barcode' or 'visual'


class FormClassifier:
    """Classifies form type using barcode or visual features"""
    
    def __init__(self, model_path: str = None, barcode_mapping: Dict[str, FormType] = None):
        """
        Args:
            model_path: Path to fine-tuned vision model (ResNet/EfficientNet)
            barcode_mapping: Dictionary mapping barcode prefixes to form types
        """
        self.barcode_mapping = barcode_mapping or {}
        self.visual_model = None
        
        # Load visual classification model if path provided
        if model_path:
            self.visual_model = self._load_model(model_path)
    
    def classify(self, image: np.ndarray, 
                 barcode_result: Optional[BarcodeResult]) -> ClassificationResult:
        """Classify form type"""
        
        # Try barcode-based classification first (most reliable)
        if barcode_result:
            form_type = self._classify_by_barcode(barcode_result)
            if form_type != FormType.UNKNOWN:
                return ClassificationResult(
                    form_type=form_type,
                    confidence=1.0,
                    method='barcode'
                )
        
        # Fall back to visual classification
        if self.visual_model:
            return self._classify_by_vision(image)
        
        # Default
        return ClassificationResult(
            form_type=FormType.UNKNOWN,
            confidence=0.0,
            method='none'
        )
    
    def _classify_by_barcode(self, barcode: BarcodeResult) -> FormType:
        """Map barcode data to form type"""
        for prefix, form_type in self.barcode_mapping.items():
            if barcode.data.startswith(prefix):
                logger.info(f"Classified as {form_type.value} via barcode")
                return form_type
        return FormType.UNKNOWN
    
    def _classify_by_vision(self, image: np.ndarray) -> ClassificationResult:
        """Classify using CNN model (placeholder implementation)"""
        # TODO: Implement actual model inference
        # This would use a fine-tuned ResNet/EfficientNet
        
        # Placeholder - would do:
        # 1. Resize/normalize image
        # 2. Run through model
        # 3. Get softmax probabilities
        # 4. Return form type with confidence
        
        logger.info("Visual classification not yet implemented")
        return ClassificationResult(
            form_type=FormType.UNKNOWN,
            confidence=0.0,
            method='visual'
        )
    
    def _load_model(self, model_path: str):
        """Load pre-trained classification model"""
        # Placeholder for model loading
        # Would load PyTorch/TensorFlow model here
        pass


# ============================================================================
# STAGE 4: FIELD EXTRACTION
# ============================================================================

@dataclass
class FieldBox:
    """Defines a field location and type for template-based extraction"""
    name: str
    x: int
    y: int
    width: int
    height: int
    field_type: str  # 'text', 'checkbox', 'date', 'number'


@dataclass
class ExtractedField:
    """Extracted field data"""
    name: str
    value: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class FieldExtractor:
    """Extracts field values using LayoutLM or template matching"""
    
    def __init__(self, use_layoutlm: bool = True, 
                 model_name: str = "microsoft/layoutlmv3-base"):
        """
        Args:
            use_layoutlm: Use LayoutLM for extraction vs template matching
            model_name: HuggingFace model name for LayoutLM
        """
        self.use_layoutlm = use_layoutlm
        self.form_templates = self._load_templates()
        
        if use_layoutlm:
            self.processor = LayoutLMv3Processor.from_pretrained(model_name, 
                                                                 apply_ocr=True)
            # Note: In production, load your fine-tuned model
            # self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            #     "path/to/your/finetuned/model"
            # )
            self.model = None  # Placeholder
    
    def extract(self, image: np.ndarray, 
                form_type: FormType) -> List[ExtractedField]:
        """Extract fields based on form type"""
        
        if self.use_layoutlm and self.model:
            return self._extract_with_layoutlm(image, form_type)
        else:
            return self._extract_with_template(image, form_type)
    
    def _extract_with_layoutlm(self, image: np.ndarray, 
                               form_type: FormType) -> List[ExtractedField]:
        """Extract using fine-tuned LayoutLMv3"""
        
        # Convert to PIL for processor
        pil_image = Image.fromarray(image)
        
        # Process image + OCR
        encoding = self.processor(pil_image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Extract fields from predictions
        # This would parse the BIO tags and extract field values
        # Placeholder implementation
        
        logger.info("LayoutLM extraction complete")
        return []
    
    def _extract_with_template(self, image: np.ndarray, 
                               form_type: FormType) -> List[ExtractedField]:
        """Extract using template matching and OCR"""
        
        template = self.form_templates.get(form_type)
        if not template:
            logger.warning(f"No template found for {form_type.value}")
            return []
        
        extracted_fields = []
        
        for field_box in template:
            # Crop field region
            y1, y2 = field_box.y, field_box.y + field_box.height
            x1, x2 = field_box.x, field_box.x + field_box.width
            field_img = image[y1:y2, x1:x2]
            
            # OCR the field
            if field_box.field_type == 'checkbox':
                value = self._detect_checkbox(field_img)
                confidence = 0.9
            else:
                # Use Tesseract for handwriting (or cloud API in production)
                config = '--psm 7'  # Single line of text
                ocr_result = pytesseract.image_to_data(
                    field_img, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                value = ' '.join([text for text in ocr_result['text'] if text.strip()])
                # Average confidence
                confs = [c for c in ocr_result['conf'] if c != -1]
                confidence = np.mean(confs) / 100 if confs else 0.0
            
            extracted_fields.append(ExtractedField(
                name=field_box.name,
                value=value,
                confidence=confidence,
                bbox=(x1, y1, x2, y2)
            ))
            
            logger.info(f"Extracted {field_box.name}: {value} (conf: {confidence:.2f})")
        
        return extracted_fields
    
    def _detect_checkbox(self, checkbox_img: np.ndarray) -> str:
        """Detect if checkbox is checked"""
        # Simple approach: count dark pixels
        total_pixels = checkbox_img.size
        dark_pixels = np.sum(checkbox_img < 128)
        fill_ratio = dark_pixels / total_pixels
        
        return "checked" if fill_ratio > 0.3 else "unchecked"
    
    def _load_templates(self) -> Dict[FormType, List[FieldBox]]:
        """Load form templates with field coordinates"""
        # In production, load from config file or database
        
        templates = {
            FormType.TYPE_A: [
                FieldBox("full_name", 100, 200, 400, 50, "text"),
                FieldBox("date_of_birth", 100, 300, 200, 50, "date"),
                FieldBox("agree_terms", 100, 400, 30, 30, "checkbox"),
            ],
            FormType.TYPE_B: [
                FieldBox("company_name", 120, 180, 450, 50, "text"),
                FieldBox("employee_id", 120, 280, 150, 50, "number"),
            ],
        }
        
        return templates


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class FormExtractionPipeline:
    """Orchestrates the complete extraction pipeline"""
    
    def __init__(self, 
                 use_layoutlm: bool = True,
                 barcode_mapping: Dict[str, FormType] = None):
        """
        Initialize pipeline components
        
        Args:
            use_layoutlm: Use LayoutLM for extraction (True) or templates (False)
            barcode_mapping: Mapping of barcode prefixes to form types
        """
        self.preprocessor = DocumentPreprocessor()
        self.barcode_detector = BarcodeDetector()
        self.classifier = FormClassifier(barcode_mapping=barcode_mapping)
        self.extractor = FieldExtractor(use_layoutlm=use_layoutlm)
    
    def process_document(self, image_path: str) -> Dict:
        """
        Process a single document through the complete pipeline
        
        Returns:
            Dictionary containing all extraction results
        """
        logger.info(f"Processing document: {image_path}")
        
        # Stage 1: Preprocess
        preprocessed_img = self.preprocessor.process(image_path)
        
        # Stage 2: Detect barcode/QR
        barcode_result = self.barcode_detector.detect(preprocessed_img)
        
        # Stage 3: Classify form type
        classification = self.classifier.classify(preprocessed_img, barcode_result)
        
        # Stage 4: Extract fields
        extracted_fields = self.extractor.extract(preprocessed_img, 
                                                  classification.form_type)
        
        # Compile results
        result = {
            'image_path': image_path,
            'barcode': {
                'detected': barcode_result is not None,
                'data': barcode_result.data if barcode_result else None,
                'type': barcode_result.type if barcode_result else None,
            },
            'classification': {
                'form_type': classification.form_type.value,
                'confidence': classification.confidence,
                'method': classification.method,
            },
            'fields': [
                {
                    'name': field.name,
                    'value': field.value,
                    'confidence': field.confidence,
                }
                for field in extracted_fields
            ]
        }
        
        logger.info(f"Extraction complete. Found {len(extracted_fields)} fields")
        return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Define barcode to form type mapping
    barcode_mapping = {
        'FORM-A-': FormType.TYPE_A,
        'FORM-B-': FormType.TYPE_B,
        'FORM-C-': FormType.TYPE_C,
    }
    
    # Initialize pipeline
    pipeline = FormExtractionPipeline(
        use_layoutlm=False,  # Set True if you have fine-tuned LayoutLM model
        barcode_mapping=barcode_mapping
    )
    
    # Process a document
    result = pipeline.process_document('/path/to/scanned/form.jpg')
    
    # Print results
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    print(f"\nForm Type: {result['classification']['form_type']}")
    print(f"Confidence: {result['classification']['confidence']:.2f}")
    print(f"Classification Method: {result['classification']['method']}")
    
    if result['barcode']['detected']:
        print(f"\nBarcode: {result['barcode']['data']} ({result['barcode']['type']})")
    
    print(f"\nExtracted Fields ({len(result['fields'])}):")
    for field in result['fields']:
        print(f"  {field['name']}: {field['value']} (confidence: {field['confidence']:.2f})")
