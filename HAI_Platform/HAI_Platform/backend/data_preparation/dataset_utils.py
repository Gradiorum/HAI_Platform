import pandas as pd

def prepare_dataset_for_inference(df, text_col, class_col=None, supp_columns=None, leading_columns=None):
    df['combined_text'] = ""
    if leading_columns:
        for col in leading_columns:
            df['combined_text'] += df[col].fillna('') + " "
    df['combined_text'] += df[text_col].fillna('')
    if supp_columns:
        for col in supp_columns:
            df['combined_text'] += " " + df[col].fillna('')

    x_data = df['combined_text'].tolist()
    y_data = df[class_col].tolist() if class_col else None
    return {"x": x_data, "y": y_data}
