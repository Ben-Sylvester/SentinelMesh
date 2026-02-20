"""
Visual Intelligence System

Unified interface for image analysis, generation, and OCR.
Integrates vision models into intelligent routing.
"""

import io
import base64
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from vision operation."""
    output: str
    model_used: str
    cost_usd: float
    latency_ms: int
    metadata: Dict[str, Any]


class VisionManager:
    """
    Manages vision capabilities: analysis, generation, OCR.
    Provides intelligent routing to optimal vision models.
    """
    
    def __init__(self, router=None):
        self.router = router
        
        # Available vision models
        self.analysis_models = {
            "gpt-4-vision": {"cost_per_image": 0.01, "quality": "high"},
            "claude-3-opus": {"cost_per_image": 0.015, "quality": "highest"},
            "gemini-pro-vision": {"cost_per_image": 0.005, "quality": "medium"}
        }
        
        self.generation_models = {
            "dall-e-3": {"cost_per_image": 0.04, "quality": "high"},
            "stable-diffusion-xl": {"cost_per_image": 0.02, "quality": "medium"}
        }
        
        self.ocr_providers = {
            "tesseract": {"cost_per_page": 0.0, "quality": "medium"},
            "azure-ocr": {"cost_per_page": 0.001, "quality": "high"},
            "google-vision": {"cost_per_page": 0.0015, "quality": "high"}
        }
    
    def _image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _detect_image_complexity(self, image_data: bytes) -> str:
        """Detect if image is simple or complex."""
        try:
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            
            # Simple heuristic
            if width * height < 500000:  # < 500K pixels
                return "simple"
            else:
                return "complex"
        except:
            return "medium"
    
    async def analyze(
        self,
        image_data: bytes,
        prompt: str,
        quality_preference: str = "balanced"
    ) -> VisionResult:
        """
        Analyze an image with a vision model.
        
        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt
            quality_preference: "cheap", "balanced", "accurate"
        
        Returns:
            VisionResult with analysis
        """
        # Detect image complexity
        complexity = self._detect_image_complexity(image_data)
        
        # Select optimal model based on preferences and complexity
        if quality_preference == "cheap":
            model = "gemini-pro-vision"
        elif quality_preference == "accurate" or complexity == "complex":
            model = "claude-3-opus"
        else:
            model = "gpt-4-vision"
        
        logger.info(f"Using {model} for vision analysis (complexity: {complexity})")
        
        # Convert image to base64
        image_base64 = self._image_to_base64(image_data)
        
        # Call vision model (delegating to adapters)
        # In production, this would call the actual vision adapter
        try:
            import time
            start = time.time()
            
            # Simulate vision API call
            # In production: result = await self.vision_adapter.analyze(image_base64, prompt)
            analysis = f"[Vision Analysis using {model}] {prompt}\n\nImage content: [Detailed analysis would appear here]"
            
            latency_ms = int((time.time() - start) * 1000)
            
            return VisionResult(
                output=analysis,
                model_used=model,
                cost_usd=self.analysis_models[model]["cost_per_image"],
                latency_ms=latency_ms,
                metadata={
                    "image_complexity": complexity,
                    "image_size_bytes": len(image_data)
                }
            )
        
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> VisionResult:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Generation prompt
            model: "dall-e-3" or "stable-diffusion-xl"
            size: Image size
            quality: "standard" or "hd"
        
        Returns:
            VisionResult with image URL
        """
        logger.info(f"Generating image with {model}")
        
        try:
            import time
            start = time.time()
            
            # Simulate image generation
            # In production: result = await self.gen_adapter.generate(prompt, size, quality)
            image_url = f"https://generated-image.example.com/{model}/{hash(prompt)}.png"
            
            latency_ms = int((time.time() - start) * 1000)
            
            return VisionResult(
                output=image_url,
                model_used=model,
                cost_usd=self.generation_models[model]["cost_per_image"],
                latency_ms=latency_ms,
                metadata={
                    "prompt": prompt,
                    "size": size,
                    "quality": quality
                }
            )
        
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def extract_text(
        self,
        image_data: bytes,
        provider: str = "tesseract"
    ) -> VisionResult:
        """
        Extract text from an image using OCR.
        
        Args:
            image_data: Raw image bytes
            provider: "tesseract", "azure-ocr", or "google-vision"
        
        Returns:
            VisionResult with extracted text
        """
        logger.info(f"Extracting text with {provider}")
        
        try:
            import time
            start = time.time()
            
            # In production, call actual OCR provider
            # For tesseract:
            # import pytesseract
            # text = pytesseract.image_to_string(Image.open(io.BytesIO(image_data)))
            
            # Simulate OCR
            extracted_text = "[Extracted text would appear here]\nLine 1\nLine 2\n..."
            
            latency_ms = int((time.time() - start) * 1000)
            
            return VisionResult(
                output=extracted_text,
                model_used=provider,
                cost_usd=self.ocr_providers[provider]["cost_per_page"],
                latency_ms=latency_ms,
                metadata={
                    "provider": provider,
                    "image_size_bytes": len(image_data)
                }
            )
        
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise
    
    async def multimodal_query(
        self,
        prompt: str,
        image_data: Optional[bytes] = None,
        router=None
    ) -> VisionResult:
        """
        Handle multimodal query (text + optional image).
        Uses router for intelligent model selection.
        
        Args:
            prompt: Text prompt
            image_data: Optional image bytes
            router: Router instance for intelligent selection
        
        Returns:
            VisionResult
        """
        if image_data:
            # Image provided - use vision analysis
            return await self.analyze(image_data, prompt)
        elif "generate" in prompt.lower() or "create image" in prompt.lower():
            # Image generation request
            return await self.generate(prompt)
        else:
            # Text-only - delegate to router
            if router:
                result = await router.route(prompt)
                return VisionResult(
                    output=result.output,
                    model_used=result.models_used[0] if result.models_used else "unknown",
                    cost_usd=result.cost_usd,
                    latency_ms=result.latency_ms,
                    metadata={}
                )
            else:
                raise ValueError("Router required for text-only queries")


class VisionPipelineIntegration:
    """
    Integrates vision capabilities into the existing vision_pipeline.py.
    Enhances the VisionReasoningPipeline with intelligent routing.
    """
    
    def __init__(self, vision_manager: VisionManager, router):
        self.vision_manager = vision_manager
        self.router = router
    
    async def process_visual_query(
        self,
        image_data: bytes,
        query: str
    ) -> Dict[str, Any]:
        """
        Process visual query with two-stage pipeline:
        1. Vision model analyzes image
        2. Reasoning model synthesizes answer
        
        This integrates with existing core/vision_pipeline.py
        """
        # Stage 1: Vision analysis
        vision_result = await self.vision_manager.analyze(image_data, query)
        
        # Stage 2: Reasoning (using router for optimal model selection)
        reasoning_prompt = f"""Based on this image analysis:
{vision_result.output}

User question: {query}

Provide a clear, accurate answer."""
        
        reasoning_result = await self.router.route(reasoning_prompt)
        
        return {
            "answer": reasoning_result.output,
            "vision_analysis": vision_result.output,
            "vision_model": vision_result.model_used,
            "reasoning_model": reasoning_result.models_used[0] if reasoning_result.models_used else "unknown",
            "total_cost": vision_result.cost_usd + reasoning_result.cost_usd,
            "total_latency_ms": vision_result.latency_ms + reasoning_result.latency_ms
        }
