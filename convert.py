import csv

def csv_to_markdown(csv_file_path, md_file_path):
    """
    Reads a CSV file and converts it into a formatted Markdown table.
    """
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)

        if not rows:
            print("The CSV file is empty.")
            return

        with open(md_file_path, mode='w', encoding='utf-8') as file:
            # 1. Write the Header Row
            headers = rows[0]
            header_line = "| " + " | ".join(headers) + " |\n"
            file.write(header_line)

            # 2. Write the Separator Row
            # This creates the standard Markdown table formatting (e.g., |---|---|)
            separator_line = "|" + "|".join(["---"] * len(headers)) + "|\n"
            file.write(separator_line)

            # 3. Write the Data Rows
            for row in rows[1:]:
                # We format each cell as a string to handle potential empty cells safely
                data_line = "| " + " | ".join([str(item) for item in row]) + " |\n"
                file.write(data_line)

        print(f"Successfully converted '{csv_file_path}' to Markdown at '{md_file_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function
if __name__ == "__main__":
    input_csv = "tokenizers.csv"
    output_md = "tokenizers_table.md"
    csv_to_markdown(input_csv, output_md)