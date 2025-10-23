# Neural Language Translator 🌐

A modern web application for neural machine translation using state-of-the-art transformer models.

## Features

- 🧠 **Neural Machine Translation**: Powered by Helsinki-NLP OPUS-MT models
- 🔍 **Auto Language Detection**: Automatically detects source language
- ⚡ **Real-time Translation**: Fast, high-quality translations
- 🎨 **Modern UI**: Clean, responsive web interface
- 🌍 **Multi-language Support**: English, Spanish, French, German, Italian, Portuguese

## Supported Language Pairs

- English ↔ Spanish, French, German, Italian, Portuguese
- Spanish, French, German, Italian, Portuguese → English

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python3 app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

3. Enter text in the input field, select languages, and click "Translate"

## API Endpoints

### Health Check
```bash
GET /health
```

### Translate Text
```bash
POST /translate
Content-Type: application/json

{
  "text": "Hello world",
  "source_lang": "en",
  "target_lang": "es"
}
```

### Detect Language
```bash
POST /detect-language
Content-Type: application/json

{
  "text": "Hello world"
}
```

### Get Supported Languages
```bash
GET /supported-languages
```

## Testing

Run the test suite to verify functionality:
```bash
python3 test_translator.py
```

## Project Structure

```
LanguageTranslation/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── test_translator.py     # Test suite
├── src/
│   ├── models/
│   │   └── translator.py  # Neural translation model
│   └── utils/
│       └── language_detector.py  # Language detection
├── templates/
│   └── index.html         # Web interface
└── static/
    ├── css/
    │   └── style.css      # Styling
    └── js/
        └── app.js         # Frontend logic
```

## Technology Stack

- **Backend**: Flask, PyTorch, Transformers (Hugging Face)
- **Models**: Helsinki-NLP OPUS-MT models
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Language Detection**: langdetect library

## Model Information

This application uses pre-trained neural machine translation models from Helsinki-NLP:
- Models are based on the Transformer architecture
- Trained on OPUS parallel text corpora
- Support for high-quality translations across multiple language pairs

## Performance Notes

- First translation may take longer as models are downloaded and loaded
- Models are cached after first use for faster subsequent translations
- GPU acceleration is automatically used if available

## License

This project is for educational and demonstration purposes.