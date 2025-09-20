## Step 1: Raw / Messy Code (Before Principles)

# Messy code – not modular, not reusable, hard to maintain
import random

numbers = [random.randint(1, 100) for _ in range(10)]
print("Generated numbers:", numbers)

# Calculate average
total = 0
for n in numbers:
    total += n
average = total / len(numbers)
print("Average:", average)

# Find max
max_num = numbers[0]
for n in numbers:
    if n > max_num:
        max_num = n
print("Max:", max_num)


"""
🔴 Problems:

No functions (not modular).

Can’t reuse logic elsewhere.

Hard to extend (e.g., adding min/median).

Not scalable (works only for small lists).

No error handling (reliability issue).

No comments/documentation.
"""

## Step 2: Refactored Code (With Principles)

import random
from typing import List

def generate_numbers(count: int, lower: int = 1, upper: int = 100) -> List[int]:
    """Generate a list of random integers."""
    return [random.randint(lower, upper) for _ in range(count)]

def calculate_average(numbers: List[int]) -> float:
    """Return the average of a list of numbers."""
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return sum(numbers) / len(numbers)

def find_max(numbers: List[int]) -> int:
    """Return the maximum number from a list."""
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return max(numbers)

if __name__ == "__main__":
    # Example workflow (can be reused in other projects)
    nums = generate_numbers(10)
    print("Generated numbers:", nums)
    print("Average:", calculate_average(nums))
    print("Max:", find_max(nums))

"""
✅ Improvements:

Modularity: Code broken into functions.

Reusability: Functions can be used in any project.

Maintainability: Easy to add min/median later.

Scalability: Can handle larger datasets (just change count).

Reliability & Quality: Error handling included.

Security & Trust: Checks against empty input.

Collaboration: Docstrings/comments make it understandable for teams.
"""

### Classroom Activity

## Step 1: Raw / Messy Pandas Code

import pandas as pd

# Load CSV
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Print average sepal length
avg = df['sepal_length'].mean()
print("Average sepal length:", avg)

# Print max petal width
mx = df['petal_width'].max()
print("Max petal width:", mx)

# Filter rows where species is setosa
print(df[df['species'] == 'setosa'].head())

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

def load_csv(url: str) -> pd.DataFrame:
    """
    Load CSV data from a URL.

    Args:
        url: URL to the CSV file

    Returns:
        DataFrame containing the CSV data, or None if loading fails
    """

    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def avg_sepal_length(df: pd.DataFrame) -> float:
    """
    Calculate average sepal length from iris dataset.

    Args:
        df: Iris dataset DataFrame

    Returns:
        Average sepal length as a float
    """
    return df['sepal_length'].mean()

def max_petal_width(df: pd.DataFrame) -> float:
    """
    Calculate maximum petal width from iris dataset.
    Args:
        df: Iris dataset DataFrame
    Returns:
        Maximum petal width as a float
    """
    return df['petal_width'].max()

def filter_species(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """
    Filter iris dataset by species.
    Args:
        df: Iris dataset DataFrame
        species: Species to filter by
    Returns:
        DataFrame containing only rows with the specified species
    """
    return df[df['species'] == species]

if __name__ == "__main__":
    df = load_csv(url=url)
    if df is not None:
        print(filter_species(df, 'setosa'))
        print(max_petal_width(df))
        print(filter_species(df, 'versicolor'))
        print(avg_sepal_length(df))


# def load_iris_data(url: str) -> Optional[pd.DataFrame]:
#     """Load iris dataset from URL with error handling.
#     Args:
#         url: URL to the CSV file
#     Returns:
#         DataFrame containing iris data, or None if loading fails
#     """
#     try:
#         return pd.read_csv(url)
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None


# def calculate_sepal_length_average(dataframe: pd.DataFrame) -> float:
#     """Calculate average sepal length from iris dataset.
#     Args:
#         dataframe: Iris dataset DataFrame
#     Returns:
#         Average sepal length
#     """
#     return dataframe['sepal_length'].mean()


# def find_max_petal_width(dataframe: pd.DataFrame) -> float:
#     """Find maximum petal width from iris dataset
#     Args:
#         dataframe: Iris dataset DataFrame
#     Returns:
#         Maximum petal width
#     """
#     return dataframe['petal_width'].max()


# def filter_by_species(dataframe: pd.DataFrame, species: str, num_rows: int = 5) -> pd.DataFrame:
#     """Filter dataset by species and return first N rows.

#     Args:
#         dataframe: Iris dataset DataFrame
#         species: Species name to filter by
#         num_rows: Number of rows to return (default: 5)

#     Returns:
#         Filtered DataFrame
#     """
#     return dataframe[dataframe['species'] == species].head(num_rows)


# def main() -> None:
#     """Main function to analyze iris dataset."""
#     # Load data
#     iris_dataframe = load_iris_data(IRIS_DATA_URL)

#     if iris_dataframe is None:
#         print("Failed to load iris dataset")
#         return

#     # Calculate and display average sepal length
#     average_sepal_length = calculate_sepal_length_average(iris_dataframe)
#     print(f"Average sepal length: {average_sepal_length}")

#     # Calculate and display maximum petal width
#     maximum_petal_width = find_max_petal_width(iris_dataframe)
#     print(f"Max petal width: {maximum_petal_width}")

#     # Filter and display setosa species
#     setosa_samples = filter_by_species(iris_dataframe, SETOSA_SPECIES)
#     print("Setosa species samples:")
#     print(setosa_samples)


# if __name__ == "__main__":
#     main()
