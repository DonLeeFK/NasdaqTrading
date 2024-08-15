import csv
import sys

def csv_to_markdown(csv_file, output_file=None):
    try:
        with open(csv_file, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            data = list(csv_reader)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create the Markdown table header
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---" for _ in headers]) + " |\n"

    # Add data rows to the Markdown table
    for row in data:
        markdown_table += "| " + " | ".join(row) + " |\n"

    if output_file:
        try:
            with open(output_file, 'w') as file:
                file.write(markdown_table)
            print(f"Markdown table has been saved to '{output_file}'")
        except IOError as e:
            print(f"Error writing to file: {e}")
    else:
        print(markdown_table)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_file> [output_markdown_file]")
    elif len(sys.argv) == 2:
        csv_to_markdown(sys.argv[1])
    else:
        csv_to_markdown(sys.argv[1], sys.argv[2])