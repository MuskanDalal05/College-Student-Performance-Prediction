# College Student Performance Measurement System

A Streamlit + Machine Learning project that lets a student enter academic details, choose Decision Tree, Random Forest, or ANN, and view a predicted performance level.

## Files
- `app.py` — Streamlit interface, model training, predictions, and evaluation.
- `college_student_performance_dataset.csv` — synthetic practice dataset (500 records).
- `requirements.txt` — Python packages.

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this ZIP and open the extracted folder in VS Code.
3. Open a terminal in that folder.
4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the application:
   ```bash
   python -m streamlit run app.py
   ```
6. Streamlit should open the app in your browser. If it does not, open the local URL shown in the terminal.

## Important notes
- Keep the CSV file in the same folder as `app.py`.
- The target is `Performance_Level` (Low, Average, High).
- `Final_Exam_Marks` is intentionally not used as an input feature to avoid target leakage.
- The included data is synthetic and intended for learning/demo use, not real academic decisions.
- Model metrics depend on this generated dataset and should not be presented as real-world performance.
