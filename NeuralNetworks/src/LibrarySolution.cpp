#include <cstdio>
#include <string>
#include <vector>

namespace Library
{
	struct TBook
	{
		std::string Title;
		std::string Author;
		int Copies;
	};

	constexpr int MAX_BOOKS = 100;
	std::vector<TBook> g_Library;

	// Function declarations
	void AddBook(const std::string& Title, const std::string& Author, int Copies);
	void DisplayBooks();
	void SearchBook(const std::string& Title);
	void CheckOutBook(const std::string& Title);
	void CheckInBook(const std::string& Title);

	void Run()
	{
		AddBook("The Great Gatsby", "F. Scott Fitzgerald", 5);
		AddBook("To Kill a Mockingbird", "Harper Lee", 3);
		DisplayBooks();
		SearchBook("To Kill a Mockingbird");
		CheckOutBook("The Great Gatsby");
		DisplayBooks();
		CheckInBook("The Great Gatsby");
		DisplayBooks();
	}

	// Function definitions
	void AddBook(const std::string& Title, const std::string& Author, const int Copies)
	{
		if (g_Library.size() < MAX_BOOKS)
		{
			g_Library.emplace_back(Title, Author, Copies);
		}
		else
		{
			printf("Library is full. Cannot add more books.\n");
		}
	}

	void DisplayBooks()
	{
		printf("Library Books:\n");

		for (const auto& Book : g_Library)
		{
			printf("Title: %s, Author: %s, Copies available: %d\n", Book.Title.c_str(), Book.Author.c_str(), Book.Copies);
		}
	}

	void SearchBook(const std::string& Title)
	{
		for (const auto& Book : g_Library)
		{
			if (Book.Title == Title)
			{
				printf("Book found:\n");
				printf("Title: %s, Author: %s, Copies available: %d\n", Book.Title.c_str(), Book.Author.c_str(), Book.Copies);

				return;
			}
		}

		printf("Book not found.\n");
	}

	void CheckOutBook(const std::string& Title)
	{
		for (auto& Book : g_Library)
		{
			if (Book.Title == Title)
			{
				if (Book.Copies > 0)
				{
					Book.Copies--;
					printf("Book checked out successfully.\n");

					return;
				}

				printf("No copies available for checkout.\n");
				return;
			}
		}

		printf("Book not found.\n");
	}

	void CheckInBook(const std::string& Title)
	{
		for (auto& Book : g_Library)
		{
			if (Book.Title == Title)
			{
				Book.Copies++;
				printf("Book checked in successfully.\n");
				return;
			}
		}

		printf("Book not found.\n");
	}
}
