"""
Document Parser Module
Extracts text and metadata from various document formats (PDF, DOCX, TXT).
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import PyPDF2
from docx import Document as DocxDocument

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text from a document with metadata.
    
    This is our fundamental data structure - every piece of text we process
    will be stored as a DocumentChunk with its associated metadata.
    """
    text: str                    # The actual text content
    page_number: Optional[int]   # Which page this came from (None for DOCX)
    chunk_index: int             # Position in the sequence of chunks
    document_name: str           # Source document filename
    document_id: str             # Unique identifier for the document
    metadata: Dict               # Additional metadata (headers, sections, etc.)
    
    def __repr__(self) -> str:
        """Readable string representation for debugging."""
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (f"DocumentChunk(doc={self.document_name}, "
                f"page={self.page_number}, chunk={self.chunk_index}, "
                f"text='{preview}')")


class DocumentParser:
    """
    Parse documents and extract text with metadata.
    
    Supports:
    - PDF files (with page-level tracking)
    - Word documents (.docx)
    - Plain text files
    
    Why we need this:
    - Different file formats require different parsing libraries
    - We need to preserve metadata (page numbers, structure)
    - Error handling for corrupted/unsupported files
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    
    def __init__(self):
        """Initialize the parser."""
        self.logger = logging.getLogger(__name__)
    
    def parse(self, file_path: str) -> Tuple[str, Dict]:
        """
        Parse a document and return (full_text, metadata).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (full_text, metadata_dict)
            
        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )
        
        # Route to appropriate parser
        if extension == '.pdf':
            return self._parse_pdf(path)
        elif extension == '.docx':
            return self._parse_docx(path)
        elif extension in {'.txt', '.md'}:
            return self._parse_text(path)
    
    def _parse_pdf(self, path: Path) -> Tuple[str, Dict]:
        """
        Parse PDF file and extract text with page information.
        
        Using PyMuPDF (fitz) for better text extraction.
        Falls back to PyPDF2 if PyMuPDF not available.
        """
        self.logger.info(f"Parsing PDF: {path.name}")
        
        # Try PyMuPDF first (better extraction)
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(path)
            num_pages = len(doc)
            
            # Extract text from each page
            pages_text = []
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean up excessive whitespace but preserve paragraphs
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text = '\n'.join(lines)
                
                pages_text.append(text)
            
            doc.close()
            
            # Combine all pages with clear separators
            full_text = '\n'.join(pages_text)
            
            # Extract metadata 
            metadata = {
                'num_pages': num_pages,
                'filename': path.name,
                'file_size': path.stat().st_size,
                'pages_text': pages_text,  # Keep page-level text
                'title': path.stem,
                'author': 'Unknown',
                'parser': 'pymupdf'
            }
            
            self.logger.info(
                f"✅ Parsed PDF with PyMuPDF: {num_pages} pages, "
                f"{len(full_text)} characters"
            )
            
            return full_text, metadata
            
        except ImportError:
            # Fallback to PyPDF2
            self.logger.warning("PyMuPDF not available, falling back to PyPDF2")
            return self._parse_pdf_pypdf2(path)
        except Exception as e:
            self.logger.error(f"❌ Error parsing PDF with PyMuPDF: {str(e)}")
            # Try PyPDF2 as fallback
            try:
                return self._parse_pdf_pypdf2(path)
            except:
                raise

    def _parse_pdf_pypdf2(self, path: Path) -> Tuple[str, Dict]:
        """
        Fallback PDF parser using PyPDF2.
        """
        self.logger.info(f"Using PyPDF2 for: {path.name}")
        
        try:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                pages_text = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Better cleaning - preserve structure
                    if text and text.strip():
                        # Remove excessive spaces but keep line breaks
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        text = '\n'.join(lines)
                        pages_text.append(text)
                    else:
                        # Empty page or failed extraction
                        pages_text.append("")
                
                # Combine all pages
                full_text = '\n\n'.join(pages_text)
                
                # Extract metadata
                metadata = {
                    'num_pages': num_pages,
                    'filename': path.name,
                    'file_size': path.stat().st_size,
                    'pages_text': pages_text,
                    'parser': 'pypdf2'
                }
                
                # Try to extract PDF metadata
                if pdf_reader.metadata:
                    metadata['title'] = pdf_reader.metadata.get('/Title', path.stem)
                    metadata['author'] = pdf_reader.metadata.get('/Author', 'Unknown')
                else:
                    metadata['title'] = path.stem
                    metadata['author'] = 'Unknown'
                
                self.logger.info(
                    f"✅ Parsed PDF with PyPDF2: {num_pages} pages, "
                    f"{len(full_text)} characters"
                )
                
                return full_text, metadata
                
        except Exception as e:
            self.logger.error(f"❌ Error parsing PDF {path.name}: {str(e)}")
            raise
    
    
    def _parse_docx(self, path: Path) -> Tuple[str, Dict]:
        """
        Parse Word document (.docx) and extract text.
        
        Why python-docx:
        - Official library for .docx files
        - Preserves paragraph structure
        - Can access headers, footers, tables
        
        Note: Only works with .docx (not old .doc format)
        """
        self.logger.info(f"Parsing DOCX: {path.name}")
        
        try:
            doc = DocxDocument(path)
            
            # Extract text from paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = '\n\n'.join(paragraphs)
            
            # Extract metadata
            metadata = {
                'num_paragraphs': len(paragraphs),
                'filename': path.name,
                'file_size': path.stat().st_size,
                'title': path.stem,
            }
            
            # Try to get document properties
            if doc.core_properties:
                metadata['author'] = doc.core_properties.author or 'Unknown'
                metadata['title'] = doc.core_properties.title or path.stem
            
            self.logger.info(
                f"Parsed DOCX: {len(paragraphs)} paragraphs, "
                f"{len(full_text)} characters"
            )
            
            return full_text, metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing DOCX {path.name}: {str(e)}")
            raise
    
    def _parse_text(self, path: Path) -> Tuple[str, Dict]:
        """
        Parse plain text file.
        
        Simple but handles encoding issues gracefully.
        """
        self.logger.info(f"Parsing text file: {path.name}")
        
        try:
            # Try UTF-8 first (most common)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    full_text = file.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 (handles most edge cases)
                with open(path, 'r', encoding='latin-1') as file:
                    full_text = file.read()
            
            metadata = {
                'filename': path.name,
                'file_size': path.stat().st_size,
                'num_lines': full_text.count('\n') + 1,
                'title': path.stem,
            }
            
            self.logger.info(
                f"✅ Parsed text file: {metadata['num_lines']} lines, "
                f"{len(full_text)} characters"
            )
            
            return full_text, metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing text file {path.name}: {str(e)}")
            raise


# Convenience function for quick parsing
def parse_document(file_path: str) -> Tuple[str, Dict]:
    """
    Quick helper to parse any supported document.
    
    Usage:
        text, metadata = parse_document("paper.pdf")
    """
    parser = DocumentParser()
    return parser.parse(file_path)