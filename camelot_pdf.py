# import streamlit as st
# import camelot
# import pandas as pd
# import tempfile
# import json
# from io import BytesIO
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os

# # ======================
# # 🔹 ENV SETUP
# # ======================
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     st.error("⚠️ GROQ_API_KEY missing in your .env file.")
#     st.stop()

# # ======================
# # 🔹 STREAMLIT UI
# # ======================
# st.set_page_config(page_title="📏 Measurement Table Extractor", layout="wide")
# st.title("📏 Textile Measurement Table Extractor")
# st.write(
#     "Upload your PDF — detects the **largest measurement table**, "
#     "extracts structured data using Groq, and lets you download as Excel."
# )

# uploaded_file = st.file_uploader("📄 Upload PDF file", type=["pdf"])

# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#         temp_pdf.write(uploaded_file.read())
#         file_path = temp_pdf.name

#     with st.spinner("🔍 Extracting tables from PDF..."):
#         try:
#             # Try lattice mode first
#             tables = camelot.read_pdf(file_path, pages="all", flavor="lattice")
#             if tables.n == 0:
#                 # Fallback to stream mode if no ruled tables detected
#                 tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
#         except Exception as e:
#             st.error(f"❌ Error reading PDF with Camelot: {e}")
#             st.stop()

#     if tables.n == 0:
#         st.error("⚠️ No tables found in the PDF.")
#         st.stop()

#     # ======================
#     # 🔹 Pick Largest Table
#     # ======================
#     largest_table = max(tables, key=lambda t: (t.shape[1], t.shape[0]))
#     df = largest_table.df

#     st.success(f"✅ Found {tables.n} tables. Using the largest one with shape {largest_table.shape}.")
#     st.subheader("📋 Extracted Table (Raw from PDF)")
#     st.dataframe(df.style.hide(axis='index'), use_container_width=True)

#     # ======================
#     # 🔹 Prepare Structured Context for Groq
#     # ======================
#     # Instead of plain text, send structured JSON to preserve column order
#     table_json = df.to_json(orient="records")

#     prompt = PromptTemplate(
#         input_variables=["context"],
#         # template=(
#         #     "You are an expert textile measurement data extractor.\n\n"
#         #     "TASK:\n"
#         #     "Analyze the provided table data in JSON format and clean it.\n"
#         #     "- Identify the table that represents measurement sizes (e.g., XS, S, M, L, XL).\n"
#         #     "- Maintain exact column order as in input.\n"
#         #     "- Ensure no value shifts between size columns.\n"
#         #     "- Keep empty cells as \"\".\n\n"
#         #     "Return the cleaned JSON array only, without explanations.\n\n"
#         #     "Input Table JSON:\n{context}"
#         # ),
#         template=(
#             "You are an expert textile measurement data extractor.\n\n"
#             "TASK:\n"
#             "1. From the provided PDF text, find all sections like “3 sizes”, “4 sizes”, “7 sizes”, etc.\n"
#             "2. Identify the one with the **largest number of sizes**.\n"
#             "3. Extract ONLY that measurement set.\n"
#             "Each measurement record may look like:\n"
#             "- A code (e.g., 'A101', 'B02')\n"
#             "- A measurement name (e.g., 'Back length from HPS', 'Across shoulder', etc.)\n"
#             "- Optional comment (e.g., 'UPDATE 29/10', 'REV 15/8', 'ADJUSTED 12/07')\n"
#             "- Optional Tolerance + (e.g., '1,50')\n"
#             "- Optional Tolerance - (e.g., '1,50')\n"
#             "    *Tolerance may also appear as 'Tol+', 'Tol-', 'TOL +', or similar — normalize all to exactly 'Tolerance +' and 'Tolerance -'.*\n"
#             "- 'Sizes'(Optional) → key-value pairs where keys are actual size names ('XS', 'S', 'M', 'L', 'XL', etc.)\n"
#             "- If a size cell has no value, use an empty string \"\".\n\n"
#             "4. Output in **pure JSON** (no explanation), structured as a list of objects:\n"
#             "Each record should include separate columns for all sizes instead of a nested 'Sizes' object.\n"
#             "Example:\n"
#             "[\n"
#             "  {{\n"
#             "    'Code': 'A101',\n"
#             "    'Measurement': 'Back length from HPS',\n"
#             "    'Comment': 'UPDATE 29/10',\n"
#             "    'Tolerance +': '1,50',\n"
#             "    'Tolerance -': '1,50',\n"
#             "    'XS': '32',\n"
#             "    'S': '34',\n"
#             "    'M': '36',\n"
#             "    'L': '38',\n"
#             "    'XL': '40',\n"
#             "    'XXL': ''\n"
#             "  }},\n"
#             "  ...\n"
#             "]\n\n"
#             "Important data handling rules:\n"
#             "- Maintain strict column alignment.\n"
#             "- If a column (like tolerance or size) is blank or not visible, output its value as an empty string \"\".\n"
#             "- **Never shift** subsequent cell values into a missing cell’s place.\n"
#             "- Keep all size names exactly as they appear in the measurement text (e.g., 'XS', 'S', 'M', 'L', 'XL', 'XXL', etc.).\n"
#             "- Some PDFs may have merged or missing tolerance values; assume empty if unclear.\n"
#             "- Preserve each row in its original logical order — do not merge or skip any line.\n\n"
#             "Rules Recap:\n"
#             "- Every key must appear in each record (even if blank).\n"
#             "- Missing values = empty string \"\".\n"
#             "- Output **pure JSON only**, no text or notes.\n"
#             "- Never skip empty tolerance or size cells.\n\n"
#             "Context:\n{context}"
#         ),
#     )

#     formatted_prompt = prompt.format(context=table_json)

#     # ======================
#     # 🔹 Groq API
#     # ======================
#     llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY)

#     with st.spinner("🧠 Cleaning and Structuring Table using Groq..."):
#         response = llm.invoke(formatted_prompt)

#     # ======================
#     # 🔹 Handle JSON Output
#     # ======================
#     try:
#         json_output = json.loads(response.content)
#         df_output = pd.DataFrame(json_output)

#         st.success("✅ Structured Measurement Data Extracted and Cleaned")
#         st.subheader("📏 Final Clean Table")
#         st.dataframe(df_output.style.hide(axis='index'), use_container_width=True)

#         # ======================
#         # 🔹 Excel Download Button
#         # ======================
#         excel_buffer = BytesIO()
#         with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
#             df_output.to_excel(writer, index=False, sheet_name="Measurements")
#         excel_buffer.seek(0)

#         st.download_button(
#             label="📥 Download Cleaned Data as Excel",
#             data=excel_buffer,
#             file_name="measurement_data_cleaned.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )

#     except json.JSONDecodeError:
#         st.warning("⚠️ Could not parse response as JSON. Showing raw output below:")
#         st.text_area("Groq Raw Output", response.content, height=400)



import streamlit as st
import camelot
import pandas as pd
import tempfile
import json
from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# ======================
# 🔹 ENV SETUP
# ======================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY missing in your .env file.")
    st.stop()

# ======================
# 🔹 STREAMLIT UI
# ======================
st.set_page_config(page_title="📏 Measurement Table Extractor", layout="wide")
st.title("📏 Textile Measurement Table Extractor")
st.write(
    "Upload your PDF — detects the **largest size measurement table**, "
    "extracts structured data using Groq, and lets you download as Excel."
)

uploaded_file = st.file_uploader("📄 Upload PDF file", type=["pdf"])

# Check if a new file was uploaded
if uploaded_file is not None:
    # If the uploaded file changed, clear the previous df_clean
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
        st.session_state.pop('df_clean', None)
        st.session_state['last_uploaded_file'] = uploaded_file.name

# if uploaded_file:
# if uploaded_file and 'df_clean' not in st.session_state:
if uploaded_file is not None and 'df_clean' not in st.session_state:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        file_path = temp_pdf.name

    # ======================
    # 🔹 Extract Tables
    # ======================
    with st.spinner("🔍 Extracting tables from PDF..."):
        try:
            tables = camelot.read_pdf(file_path, pages="all", flavor="lattice", split_text=True)
            if tables.n == 0:
                tables = camelot.read_pdf(file_path, pages="all", flavor="stream", split_text=True)
        except Exception as e:
            st.error(f"❌ Error reading PDF with Camelot: {e}")
            st.stop()

    if tables.n == 0:
        st.error("⚠️ No tables found in the PDF.")
        st.stop()

    st.success(f"✅ Found {tables.n} tables in the PDF.")

    # ======================
    # 🔹 Prepare Table Summaries (small for LLM)
    # ======================
    table_summaries = []
    for i, t in enumerate(tables):
        df_temp = t.df
        sample_rows = df_temp.head(5).values.tolist()  # only first 5 rows to reduce token size
        table_summaries.append({
            "table_index": i + 1,
            "shape": t.shape,
            "header_sample": sample_rows
        })

    summary_json = json.dumps(table_summaries)

    # ======================
    # 🔹 Step 1: Identify Largest Size Table
    # ======================
    st.info("🧠 Asking Groq to identify the largest size measurement table...")

    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY)

    prompt_identify_table = PromptTemplate(
        input_variables=["context"],
        template=(
            "You are a textile measurement table expert.\n\n"
            "You are given multiple tables extracted from a PDF. Each table has a sample of first 5 rows and a shape.\n"
            "TASK:\n"
            "1. Identify the table which contains the measurement with the **largest number of sizes** (e.g., 3, 4, 7 sizes).\n"
            "2. Only output the **table_index** (integer) of that table.\n\n"
            "Input:\n{context}\n"
        )
    )

    formatted_prompt = prompt_identify_table.format(context=summary_json)

    try:
        response = llm.invoke(formatted_prompt)
        table_index = int(response.content.strip())
        #st.success(f"✅ Largest size measurement table identified: Table {table_index}")
    except Exception as e:
        st.error(f"❌ Groq could not identify largest size table: {e}")
        st.stop()

    # ======================
    # 🔹 Extract that Table
    # ======================
    selected_table = tables[table_index - 1].df
    #st.subheader("📋 Selected Table (Raw from PDF)")
    #st.dataframe(selected_table.style.hide(axis='index'), use_container_width=True)

    # ======================
    # 🔹 Step 2: Clean & Structure Table
    # ======================
    st.info("🧠 Cleaning and structuring the selected table using Groq...")

    table_json = selected_table.to_json(orient="records")

    prompt_clean_table = PromptTemplate(
        input_variables=["context"],
        template=(
            "You are an expert textile measurement data extractor.\n\n"
            "TASK:\n"
            "Clean and structure the measurement table in JSON.\n"
            "1. Extract ONLY that measurement set.\n"
            "Each measurement record may look like:\n"
            "- A code (e.g., 'A101', 'B02')\n"
            "- A measurement name (e.g., 'Back length from HPS', 'Across shoulder', etc.)\n"
            "- Optional comment (e.g., 'UPDATE 29/10', 'REV 15/8', 'ADJUSTED 12/07')\n"
            "- Optional Tolerance + (e.g., '1,50')\n"
            "- Optional Tolerance - (e.g., '1,50')\n"
            "    *Tolerance may also appear as 'Tol+', 'Tol-', 'TOL +', or similar — normalize all to exactly 'Tolerance +' and 'Tolerance -'.*\n"
            "- 'Sizes'(Optional) → key-value pairs where keys are actual size names ('XS', 'S', 'M', 'L', 'XL', etc.) or ('32','34','36','38' etc.) , use like in table\n"
            "- If a size cell has no value, use an empty string \"\".\n\n"
            "2. Output in **pure JSON** (no explanation), structured as a list of objects:\n"
            "Each record should include separate columns for all sizes instead of a nested 'Sizes' object.\n"
            "Example:\n"
            "[\n"
            "  {{\n"
            "    'Code': 'A101',\n"
            "    'Measurement': 'Back length from HPS',\n"
            "    'Comment': 'UPDATE 29/10',\n"
            "    'Tolerance +': '1,50',\n"
            "    'Tolerance -': '1,50',\n"
            "    'XS': '32',\n"
            "    'S': '34',\n"
            "    'M': '36',\n"
            "    'L': '38',\n"
            "    'XL': '40',\n"
            "    'XXL': ''\n"
            "  }},\n"
            "  ...\n"
            "]\n\n"
            "Important data handling rules:\n"
            "- Maintain strict column alignment.\n"
            "- If a column (like tolerance or size) is blank or not visible, output its value as an empty string \"\".\n"
            "- **Never shift** subsequent cell values into a missing cell’s place.\n"
            "- Keep all size names exactly as they appear in the measurement text (e.g., 'XS', 'S', 'M', 'L', 'XL', 'XXL', etc.).\n"
            "- Some PDFs may have merged or missing tolerance values; assume empty if unclear.\n"
            "- Preserve each row in its original logical order — do not merge or skip any line.\n\n"
            "Rules Recap:\n"
            "- Every key must appear in each record (even if blank).\n"
            "- Missing values = empty string \"\".\n"
            "- Output **pure JSON only**, no text or notes.\n"
            "- Never skip empty tolerance or size cells.\n\n"
            "Input Table JSON:\n{context}"
        )
    )

    
    formatted_prompt_clean = prompt_clean_table.format(context=table_json)

    with st.spinner("🧹 Structuring table with Groq..."):
        try:
            response_clean = llm.invoke(formatted_prompt_clean)
            structured_json = json.loads(response_clean.content)
            df_clean = pd.DataFrame(structured_json)


            # Save in session state to prevent recomputation
            st.session_state['df_clean'] = df_clean

        except Exception as e:
            st.error(f"❌ Could not process table with Groq: {e}")
            st.text_area("Groq Raw Output", response_clean.content, height=400)

if 'df_clean' in st.session_state:
    df_clean = st.session_state['df_clean']    
        
    st.success("✅ Structured Measurement Data Extracted")
    st.subheader("📏 Final Clean Table")
    st.dataframe(df_clean.style.hide(axis='index'), use_container_width=True)

    # ======================
    # 🔹 Excel Download Button
    # ======================
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="Measurements")
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Cleaned Data as Excel",
        data=excel_buffer,
        file_name="measurement_data_cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


