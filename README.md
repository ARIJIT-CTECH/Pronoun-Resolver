# 🔍 Pronoun Resolution Analyzer

A comprehensive Streamlit web application that compares **heuristic-based** vs **deep learning** approaches for pronoun coreference resolution.

## 🎯 Features

- **Dual Approach Comparison**: Compare rule-based and neural network methods side-by-side
- **Interactive UI**: Clean, modern Streamlit interface with real-time analysis
- **Visual Highlighting**: See pronouns resolved and highlighted in context
- **Performance Metrics**: Compare processing speed and accuracy between approaches
- **Export Options**: Download results in multiple formats (CSV, JSON, TXT)
- **Sample Texts**: Built-in examples for quick testing

## 🧠 Approaches Implemented

### 1. Heuristic-Based Resolution
- **Nearest noun matching**
- **Gender agreement** (he/she/they)
- **Number agreement** (singular/plural)
- **Subject preference** (subjects over objects)
- **Named entity priority**

### 2. Deep Learning-Based Resolution
- **Transformer models** (AllenAI Coref-SpanBERT)
- **Neural coreference clustering**
- **Context-aware resolution**
- **Fallback to spaCy** when models unavailable

## 🚀 Quick Start

### Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Usage

1. **Enter or paste text** containing pronouns
2. **Select approaches** to compare (heuristic, deep learning, or both)
3. **Click "Resolve Pronouns"** to analyze
4. **Review results** with highlighted text and comparison metrics
5. **Export results** for further analysis

## 📁 Project Structure

```
pronoun-resolution-analyzer/
├── app.py                 # Main Streamlit application
├── heuristic.py           # Rule-based pronoun resolver
├── deep_learning.py      # Neural network-based resolver
├── utils.py              # Utility functions and helpers
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 📦 Dependencies

- **streamlit** >= 1.28.0 - Web framework
- **spacy** >= 3.6.0 - NLP processing
- **nltk** >= 3.8.1 - Text processing
- **transformers** >= 4.35.0 - Transformer models
- **torch** >= 2.0.0 - Deep learning framework
- **numpy** >= 1.24.0 - Numerical computing
- **pandas** >= 2.0.0 - Data manipulation
- **plotly** >= 5.15.0 - Visualization
- **tqdm** >= 4.65.0 - Progress bars

## 🎨 UI Features

- **Responsive design** with wide layout
- **Interactive sidebar** for configuration
- **Sample text buttons** for quick testing
- **Real-time progress indicators**
- **Color-coded results** (heuristic: red, deep learning: teal)
- **Expandable sections** for detailed views
- **Metric cards** for performance comparison

## 📊 Example Usage

### Input Text:
```
John went to the store. He bought milk. Then he met Sarah and she greeted him.
```

### Heuristic Results:
- **he** → John
- **she** → Sarah

### Deep Learning Results:
- **Cluster 1**: John, He, he
- **Cluster 2**: Sarah, she

### Comparison Metrics:
- **Agreement Rate**: 100%
- **Processing Time**: Heuristic (50ms) vs Deep Learning (200ms)
- **Resolution Count**: 3 pronouns found

## 🔧 Configuration Options

### Model Settings
- Toggle heuristic approach on/off
- Toggle deep learning approach on/off
- Adjust display preferences

### Display Options
- Show/hide coreference clusters
- Show/hide comparison tables
- Show/hide performance metrics
- Enable/disable text highlighting

## ⚡ Performance

- **Heuristic Approach**: Fast (~10-50ms), rule-based
- **Deep Learning Approach**: Slower (~100-500ms), more accurate
- **Model Caching**: Models loaded once per session
- **Memory Efficient**: Optimized for typical text lengths

## 🛠️ Technical Details

### Heuristic Algorithm
1. **Entity Extraction**: Using spaCy/NLTK for NER and POS tagging
2. **Candidate Scoring**: Distance, gender, number, and position features
3. **Resolution**: Select highest-scoring antecedent for each pronoun

### Deep Learning Algorithm
1. **Model Loading**: AllenAI Coref-SpanBERT via HuggingFace
2. **Text Processing**: Tokenization and encoding
3. **Coreference Clustering**: Neural network predictions
4. **Fallback**: spaCy-based resolution when models unavailable

## 🔍 Error Handling

- **Input Validation**: Checks for minimum text length and pronoun presence
- **Model Fallbacks**: Graceful degradation when models unavailable
- **Exception Handling**: User-friendly error messages
- **Resource Management**: Automatic cleanup and caching

## 📈 Advanced Features

### Export Formats
- **CSV**: Tabular comparison data
- **JSON**: Complete analysis results
- **TXT**: Human-readable report

### Metrics Tracked
- **Resolution Count**: Number of pronouns resolved
- **Agreement Rate**: Percentage of matching results
- **Processing Time**: Performance comparison
- **Speedup Factor**: Relative performance

## 🚀 Future Enhancements

- **Additional Models**: Support for more transformer models
- **Batch Processing**: Analyze multiple texts simultaneously
- **Custom Rules**: User-defined heuristic rules
- **Performance Graphs**: Historical performance tracking
- **API Integration**: REST API for programmatic access

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **AllenAI** for the Coref-SpanBERT model
- **spaCy** for NLP processing tools
- **Streamlit** for the web framework
- **HuggingFace** for model hosting

## 📞 Support

For issues, questions, or feature requests:
1. Check the **Issues** section
2. Review existing **documentation**
3. Create a **new issue** with detailed information

---

**Built with ❤️ using Python, Streamlit, and Modern NLP Techniques**
