# Business Intelligence Prompts

This directory contains prompts for generating comprehensive business intelligence insights from call center data across any industry.

## Files Overview

### 1. `bi_analysis_system_prompt.txt`
The system prompt that defines the AI's role as a Business Intelligence Analyst specialized in call center operations. This prompt:
- Establishes core capabilities and analysis framework
- Defines output requirements and visualization specifications
- Ensures industry-agnostic approach
- Specifies JSON structure requirements

### 2. `bi_analysis_user_prompt.txt`
The user prompt template for requesting business intelligence analysis. This template:
- Accepts company context and call data as variables
- Defines specific analysis requirements
- Outlines focus areas and visualization needs
- Ensures comprehensive output format

### 3. `bi_analysis_json_schema.json` ✨ **ACTIVELY USED**
The complete JSON schema that defines the exact structure for business intelligence output. This schema is:
- **Automatically loaded and included in system prompts** for better AI guidance
- **Used for response validation** to ensure output quality
- **Provides concrete examples** of expected data structures
- **Improves response consistency** across different AI models

**Key Sections Included:**
- Analysis metadata and data summaries
- Executive summary with key findings
- Customer experience insights
- Agent performance analysis
- Operational efficiency metrics
- Business intelligence and risk assessment
- Actionable recommendations
- Visualization dashboard specifications
- Benchmarks and targets

## Usage

### In API Endpoints
These prompts are designed to be used in OpenAI API calls where:
- `bi_analysis_system_prompt.txt` is used as the system message
- `bi_analysis_user_prompt.txt` is used as the user message with variable substitution
- `bi_analysis_json_schema.json` is **automatically loaded and appended** to the system prompt

### Variable Substitution
The user prompt template uses these variables:
- `{company_name}` - Name of the company being analyzed
- `{industry}` - Industry type (if known)
- `{start_date}` - Analysis period start date
- `{end_date}` - Analysis period end date
- `{total_calls}` - Total number of calls in the dataset
- `{call_data_json}` - The actual call data in JSON format

### Example Implementation
```python
from src.utils.llm import LLM

# Single model analysis (schema automatically included)
llm = LLM(model="gpt-4-turbo")
result = llm.analyze_business_intelligence(
    call_data=your_call_data,
    company_name="Shell Club",
    industry="Petroleum/Loyalty Program",
    user_info={"user_id": "123", "username": "analyst"}
)

# Model comparison (schema validation included)
comparison = llm.compare_bi_analysis(
    call_data=your_call_data,
    company_name="Shell Club",
    industry="Petroleum/Loyalty Program",
    openai_model="gpt-4-turbo",
    gemini_model="gemini-2.0-flash"
)
```

## Schema Integration Features

### 🔄 **Automatic Schema Loading**
The LLM implementation automatically:
1. Loads the JSON schema from `bi_analysis_json_schema.json`
2. Appends it to the system prompt with formatting instructions
3. Ensures the AI has a concrete example of expected output

### ✅ **Response Validation**
Each analysis response is automatically validated against the schema:
- **Structure completeness**: Checks for all required sections
- **Data type validation**: Ensures correct data types (numbers, arrays, objects)
- **Field validation**: Verifies required fields are present
- **Score validation**: Validates score ranges (0-100 for satisfaction scores)
- **Visualization structure**: Ensures visualization arrays are properly formatted

### 📊 **Quality Metrics**
The validation provides:
- **Completeness score**: Percentage of required sections present
- **Validation status**: Pass/fail for schema compliance
- **Issue tracking**: Detailed list of any structural problems
- **Quality comparison**: Schema compliance comparison between models

## Key Features

### Universal Applicability
- Works across any industry or company type
- Focuses on universal call center challenges
- Provides industry-agnostic recommendations

### Comprehensive Analysis
- Customer experience and satisfaction
- Agent performance and coaching needs
- Operational efficiency optimization
- Business intelligence and strategic insights
- Risk assessment and early warnings

### Enhanced Quality Assurance
- **Schema-guided responses**: AI receives exact output format requirements
- **Automatic validation**: Every response is checked against the schema
- **Consistency improvement**: Reduces parsing errors and structural inconsistencies
- **Model comparison**: Schema compliance becomes a quality metric

### Actionable Outputs
- Specific recommendations with priority levels
- Visualization specifications for dashboards
- Performance benchmarks and targets
- Implementation timelines and resource requirements

### Structured JSON Output
- Consistent format for easy parsing
- Comprehensive coverage of all analysis areas
- Ready for dashboard and reporting integration
- **Schema-validated** for guaranteed structure compliance

## Integration Notes

These prompts are designed to integrate with:
- The existing AI insights API endpoints
- Dashboard visualization systems
- Business intelligence reporting tools
- Real-time monitoring and alerting systems

The output format ensures compatibility with modern BI tools and data visualization frameworks, with **guaranteed schema compliance** for reliable integration.

## Schema Validation Benefits

1. **Improved Response Quality**: AI gets concrete examples of expected output
2. **Reduced Parsing Errors**: Schema validation catches structural issues
3. **Model Performance Comparison**: Schema compliance becomes a quality metric
4. **Debugging Support**: Detailed validation reports help identify issues
5. **Integration Reliability**: Guaranteed structure for downstream systems 