# Systems Thinking Research Approach: Understanding and Implementation

## Overview
This document explains how researchers can capture public firms' level of systems thinking by analyzing their annual reports. Systems thinking is a way of looking at complex problems by considering the relationships between different parts rather than focusing on individual elements. Our approach uses artificial intelligence (AI) in a supervised manner, where we train the AI to extract, process, and classify text from reports.

## Table of Contents
1. [How the Model Works](#how-the-model-works)
2. [Step 1: Collecting and Processing Reports](#step-1-collecting-and-processing-reports)
3. [Step 2: Splitting the Text into Meaningful Sections](#step-2-splitting-the-text-into-meaningful-sections)
4. [Step 3: Converting Text into Meaningful Representations](#step-3-converting-text-into-meaningful-representations)
5. [Step 4: Detecting and Categorizing Systems Thinking](#step-4-detecting-and-categorizing-systems-thinking)
6. [Step 5: Evaluating AI Confidence and Large-Scale Processing](#step-5-evaluating-ai-confidence-and-large-scale-processing)
7. [Step 6: Enhancing Classification with Fine-Tuning and RAG](#step-6-enhancing-classification-with-fine-tuning-and-rag)
8. [Step 7: Measuring Performance and Accuracy](#step-7-measuring-performance-and-accuracy)
9. [Step 8: Refining AI Decisions with Human Feedback](#step-8-refining-ai-decisions-with-human-feedback)
10. [Cloud Deployment](#cloud-deployment)

## How the Model Works
We break down the annual report text into paragraphs and identify sections that reflect systems thinking. This is done by training AI to replicate our hand-coding process, which categorizes statements into different theoretical subdimensions of systems thinking:

- **Purpose**: Alignment with long-term objectives and vision
- **Macro Issue Why**: Explains why a macro issue is important for the firm, zooming out to a broader system level
- **Macro Issue How**: Details specific initiatives addressing system-level problems, zooming out
- **Micro Issue Why**: Explains why a micro issue is important, zooming in to internal organizational concerns
- **Micro Issue How**: Discusses specific initiatives to tackle internal system problems, zooming in
- **Collaboration**: Partnerships and cooperative strategies
- **Agency**: Actions the company takes to transform the system, reflecting a proactive stance

## Step 1: Collecting and Processing Reports
The first step involves gathering and preparing company reports for analysis. The initial dataset is intentionally limited to 22 annual reports, allowing for meticulous hand-coding before scaling the analysis. Since these reports come in various formats, usually as PDFs, we first standardize them:

1. **Text Extraction**: Extract text from PDFs, with special handling for:
   - Digitally structured documents
   - Scanned documents requiring OCR
   
2. **Data Cleaning**:
   - Remove unnecessary symbols
   - Correct formatting errors
   - Standardize text structure

### Example: Data Cleaning Process

```plaintext
Raw Extracted Text:
Company Initiative: pep+ $$$
Sustainability Report 2022 ***
Key Metrics & Goals !!!

Cleaned Data:
Company Initiative: pep+
Sustainability Report 2022
Key Performance Metrics
```

## Step 2: Splitting the Text into Meaningful Sections
After extracting the raw text, we refine the segmentation process to ensure that information is processed accurately. Since paragraphs serve as our unit of analysis, we apply strict filtering criteria to maintain high-quality data extraction.

### 2.1 Paragraph Segmentation and Length Criteria
- **Minimum Requirements**: Each paragraph must contain at least 1 sentence
- **Character Limits**: 
  - Lower bound: 50 characters (filters out headers/titles)
  - Upper bound: 2,000 characters (prevents parsing errors)
- **Sentence Limits**: Maximum 99 sentences per paragraph

### 2.2 Content Filtering
We exclude the following to ensure only meaningful narrative text is processed:
- Tables and structured numerical data
- Legal disclaimers and footnotes
- Auditor's reports and financial reviews
- Images and non-text elements

## Step 3: Converting Text into Meaningful Representations
Before classification, the text must be converted into a structured format that AI can process. This involves two key steps:

### 3.1 Tokenization
- Breaks text into smaller units (tokens)
- Ensures consistent processing of word variations
- Uses pre-trained AutoTokenizer for efficient encoding

### 3.2 Embeddings
- Transforms text into high-dimensional numerical representations
- Captures semantic relationships between words
- Uses OpenAI's text-embedding-3-large model

### Example: Text Processing Pipeline
```plaintext
Original Text: "Our company prioritizes sustainability through renewable energy projects."

Tokenized: ["our", "company", "prioritizes", "sustainability", "through", "renewable", "energy", "projects"]

Embeddings: [0.23, -0.56, 0.89, -0.12, 0.75, 0.44, -0.98, 0.65]
```

## Step 4: Detecting and Categorizing Systems Thinking
The AI follows a two-step hierarchical classification process to determine if and how a paragraph exhibits systems thinking.

### 4.1 Two-Step Classification Process
1. **Initial Detection**: Classifies paragraphs as either systems thinking (ST) or not (ST: No)
2. **Subdimension Classification**: For ST paragraphs, assigns one or more of the eight subdimensions

### 4.2 Classification Workflow
1. **Binary Classification**: Determines if paragraph contains systems thinking
2. **Subdimension Assignment**: Uses LLM-based prompt engineering for detailed categorization
3. **Context Integration**: Feeds previously labeled paragraphs into LLM prompt for consistency

### Example: Classification Results
```plaintext
Input Text: "We strive to make a positive impact on the environment and society while delivering long-term value for our stakeholders."

Step 1: Systems Thinking? Yes (95% confidence)
Step 2: Subdimensions: Purpose, Macro Issue How
```

## Step 5: Evaluating AI Confidence and Large-Scale Processing
The system implements robust confidence scoring and efficient processing mechanisms for large-scale analysis.

### 5.1 Confidence Scoring
- Score Range: 0 to 1
- High Confidence: ≥ 0.85
- Scores reflect similarity to expert-labeled examples

### 5.2 Large-Scale Processing
- Parallel processing for multiple paragraphs
- Batch classification for similar content
- Confidence-based prioritization for review

### Example: Confidence Scoring
```plaintext
Text: "We manage our ingredient risks through multiple geographies and suppliers, while developing sustainable farming practices globally."
Classification: Systems Thinking
Confidence: 0.95
Subdimensions: Micro Issue How, Agency
```

## Step 6: Enhancing Classification with Fine-Tuning and RAG
The model combines fine-tuning and Retrieval-Augmented Generation (RAG) to improve classification accuracy.

### 6.1 Fine-Tuning
- Trains on expert hand-coded examples
- Builds conceptual foundation of systems thinking
- Enables generalization to new language patterns

### 6.2 RAG Implementation
- Retrieves similar, expert-coded paragraphs at runtime
- Adapts to contextual differences in language
- Improves classification accuracy through context

### 6.3 Combined Benefits
- Leverages generalized patterns from fine-tuning
- Incorporates specific examples through RAG
- Maintains consistency while allowing contextual adaptation

## Step 7: Measuring Performance and Accuracy
The system employs comprehensive metrics to evaluate classification performance.

### 7.1 Key Performance Metrics
| Metric | Formula | Target Range |
|--------|----------|--------------|
| Accuracy | (TP + TN) / Total | 88-92% |
| Precision | TP / (TP + FP) | >85% |
| Recall | TP / (TP + FN) | >85% |
| F1-Score | 2 × (P × R) / (P + R) | >85% |

### 7.2 Performance Monitoring
- Regular validation against expert-labeled data
- Continuous tracking of classification accuracy
- Periodic model retraining and optimization

## Step 8: Refining AI Decisions with Human Feedback
The system incorporates Reinforcement Learning from Human Feedback (RLHF) for continuous improvement.

### 8.1 RLHF Process
1. Expert review of AI classifications
2. Structured feedback collection
3. Model policy updates based on feedback

### 8.2 Feedback Integration
- Updates classification policies
- Refines decision thresholds
- Improves subdimension alignment

## Cloud Deployment
The system is deployed on AWS infrastructure for scalability and reliability.

### Key Components
- **AWS S3**: Document storage and management
- **AWS SageMaker**: Model hosting and inference
- **Amazon OpenSearch**: Efficient text search and retrieval
- **AWS Lambda**: Automated processing workflows

### Architecture Benefits
- Scalable processing capacity
- Reliable data storage and backup
- Efficient resource utilization
- Automated workflow management
