#pragma once

class Matrix
{
public:
	Matrix();
	explicit Matrix(size_t rowCount, size_t columnCount, bool randomise = false);
	explicit Matrix(const std::vector<std::vector<float>>& rowsOfColumns);
	Matrix(const Matrix& other);
	Matrix(Matrix&& other) noexcept;
	Matrix& operator=(const Matrix& rhs);
	Matrix& operator=(Matrix&& rhs) noexcept;
	~Matrix();

	[[nodiscard]] Matrix SubMatrix(size_t rowOffset, size_t columnOffset, size_t subMatrixRowCount, size_t subMatrixColumnCount) const;
	[[nodiscard]] float& At(size_t rowIndex, size_t columnIndex);
	[[nodiscard]] float At(size_t rowIndex, size_t columnIndex) const;

	[[nodiscard]] float* GetData() const;
	[[nodiscard]] size_t GetRowCount() const;
	[[nodiscard]] size_t GetColumnCount() const;

	void Dot(const Matrix& rhs, Matrix& resultOut) const;
	void Sum(const Matrix& rhs, Matrix& resultOut) const;
	void Sum(const Matrix& rhs);
	void Activate();

	void PrintValues(const char* name = "None", const char* formatSpecifier = "%f") const;

	Matrix operator*(const Matrix& rhs) const;
	Matrix operator+(const Matrix& rhs) const;
	Matrix& operator+=(const Matrix& rhs);

private:
	size_t MatrixIdentifier;
	size_t RowCount;
	size_t ColumnCount;
	float* Data;

	static size_t MatrixIdCounter;
};
