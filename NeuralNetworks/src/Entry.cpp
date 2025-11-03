// Precompiled headers
#include "Pch.h"

#include "TinyDnnAdder.h"

// Entry point and simple text menu used to pick and run small demo models.
// The menu is intentionally lightweight: it displays a list of demos, shows a
// short preview/description for the selected demo, and runs the demo callback
// when the user confirms.
namespace MenuInternal
{
	// Represents a single menu entry.
	struct MenuItem
	{
		int Id;
		std::string Name;
		std::string Description;
		std::string ArchitectureInfo;
		std::function<void()> Callback;
	};

	// Minimal console menu.
	class Menu
	{
		std::vector<MenuItem> Items;
		int NextId = 1; // next id assigned to a newly added item

	public:
		Menu() = default;

		// Register a new menu entry.
		void AddItem(const std::string_view name, const std::string_view description, const std::string_view architectureInfo, const std::function<void()>& callback)
		{
			Items.emplace_back(NextId++, std::string(name), std::string(description), std::string(architectureInfo), callback);
		}

		// Display compact menu: one line per item showing id and name.
		void ShowSummary() const
		{
			for (const auto& it : Items)
			{
				// "id) title - short description"
				std::println("{}) {} - {}", it.Id, it.Name, it.Description);
			}

			std::println("q) Quit");
			std::printnl();
			std::print("Select an option number to preview and run: ");
		}

		// Display a preview for a single menu item and prompt for confirmation.
		void ShowPreview(const MenuItem& it) const
		{
			std::printnl();
			std::println("--- Preview: {} ---", it.Name);

			if (!it.ArchitectureInfo.empty())
			{
				std::println("Architecture: {}", it.ArchitectureInfo);
			}

			if (!it.Description.empty())
			{
				std::println("Description: {}", it.Description);
			}

			std::printnl();
			std::print("Run this model? (y/n): ");
		}

		// Read a single line.
		static std::string ReadLine()
		{
			std::string s;
			std::getline(std::cin, s);
			return s;
		}

		// Parse a user's selection. Accepts a numeric selection or 'q' to quit.
		static std::optional<int> ParseSelection(const std::string& input)
		{
			if (input.empty())
			{
				return std::nullopt;
			}

			if (input == "q" || input == "Q")
			{
				return std::nullopt;
			}

			// Try parse integer.
			try
			{
				size_t pos = 0;
				int value = std::stoi(input, &pos);
				if (pos == input.size())
				{
					return value;
				}
			}
			catch (...)
			{
				// fallthrough
			}

			return std::nullopt;
		}

		void Run()
		{
			while (true)
			{
				// 1) Show available items and prompt for selection.
				ShowSummary();

				std::string selection = ReadLine();
				if (selection == "q" || selection == "Q")
				{
					std::println("Exiting.");
					return;
				}

				// 2) Parse the input into an item id.
				auto maybeId = ParseSelection(selection);
				if (!maybeId.has_value())
				{
					std::println("Invalid selection. Try again.");
					continue;
				}

				int chosenId = maybeId.value();

				// Find the matching item.
				auto item = std::ranges::find_if(Items.begin(), Items.end(), [chosenId](const MenuItem& m)
				{
					return m.Id == chosenId;
				});

				if (item == Items.end())
				{
					std::println("No such menu item. Try again.");
					continue;
				}

				// 3) Show a preview and ask the user to confirm running the demo
				ShowPreview(*item);
				std::string confirm = ReadLine();
				if (confirm == "y" || confirm == "Y")
				{
					std::printnl();
					std::println("Running \"{}\"... (output will be shown below)", item->Name);
					std::printnl();

					// 4) Execute the callback. Exceptions are caught to avoid crashing the menu.
					try
					{
						item->Callback();
					}
					catch (const std::exception& ex)
					{
						std::println("Error while running: {}", ex.what());
					}
					catch (...)
					{
						std::println("Unknown error while running.");
					}

					std::printnl();
					std::println("--- Run complete ---");
					std::print("Press Enter to return to the menu...");
					ReadLine();
				}
				else
				{
					std::println("Cancelled. Returning to menu.");
				}
			}
		}
	};
}

int main(int argc, char* argv[])
{
	MenuInternal::Menu menu;

	menu.AddItem("TinyDnnAdder (4-bit)",
	             "Train and preview a small 4-bit adder network using tiny-dnn.",
	             "Input(9) -> Hidden(32)(ReLU) -> Hidden(16)(ReLU) -> Output(5)(Sigmoid)",
	             []
	             {
		             TinyDnnAdder::Run();
	             });

	menu.Run();

	return 0;
}
