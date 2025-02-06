#pragma once

#include <vector>

class Matrix
{
public:
	Matrix();
	explicit Matrix(size_t RowCount, size_t ColumnCount, bool Randomise = false);
	explicit Matrix(const std::vector<std::vector<float>>& RowsOfColumns);
	Matrix(const Matrix& Source);
	Matrix& operator=(const Matrix& Rhs);
	Matrix& operator=(Matrix&& Rhs) noexcept;
	~Matrix();

	Matrix SubMatrix(size_t RowOffset, size_t ColumnOffset, size_t SubMatrixRowCount, size_t SubMatrixColumnCount) const;
	float& At(size_t RowIndex, size_t ColumnIndex);
	float At(size_t RowIndex, size_t ColumnIndex) const;

	float* GetData() const;
	size_t GetRowCount() const;
	size_t GetColumnCount() const;

	void Dot(const Matrix& Rhs, Matrix& ResultOut) const;
	void Sum(const Matrix& Rhs, const Matrix& ResultOut) const;
	void Sum(const Matrix& Rhs);
	void Activate() const;

	void PrintValues(const char* Name = "None", const char* FormatSpecifier = "%f") const;

	Matrix operator*(const Matrix& Rhs) const;
	Matrix operator+(const Matrix& Rhs) const;
	Matrix& operator+=(const Matrix& Rhs);

private:
	size_t MatrixIdentifier;
	size_t RowCount;
	size_t ColumnCount;
	float* Data;
};
