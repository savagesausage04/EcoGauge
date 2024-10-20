import reflex as rx

colors = {
    "background": "#E6E6FA",  # Lavender purple
    "text": "#333333",        # Dark gray for main text
    "accent_blue": "#007acc", # Blue for header background and accents
    "accent_green": "#A8E6CE",# Light green for some text and button
    "input_bg": "#f0f0f0",    # Light gray for input background
    "input_text": "#333333",  # Dark gray for input text
    "hover_teal": "#2BB673",  # Darker teal for button hover
}

button_hover_effects = """
    <style>
        .hover-button {
            background-color: #A8E6CE;
            transition: background-color 0.3s;
        }

        .hover-button:hover {
            background-color: #2BB673;
        }
    </style>
"""

# Adjust the duration for the scrolling animation to make it slower
anime_duration = 60  # seconds

fruits = [
    "Apple", "Apricot", "Avocado", "Banana", "Bilberry", "Blackberry", "Blackcurrant",
    "Blueberry", "Boysenberry", "Currant", "Cherry", "Cherimoya", "Chico fruit",
    "Cloudberry", "Coconut", "Cranberry", "Cucumber", "Custard apple", "Damson", "Date"
]

reverted_scrolling_text = f'''
<style>
@keyframes scrolling-text {{
  0% {{ transform: translateX(100%); }}
  100% {{ transform: translateX(-100%); }}
}}
.scrolling-container {{
  overflow: hidden;
  white-space: nowrap;
  width: 100%;
  background: rgba(0, 122, 204, 0.1);
  padding: 10px 0;
  position: absolute;
  bottom: 0;
  left: 0;
}}
.scrolling-text {{
  display: inline-block;
  animation: scrolling-text {anime_duration}s linear infinite;
}}
</style>
<div class="scrolling-container">
  <div class="scrolling-text">
    {' &nbsp;&nbsp;&nbsp; '.join(fruits)} &nbsp;&nbsp;&nbsp;
    {' &nbsp;&nbsp;&nbsp; '.join(fruits)}
  </div>
</div>
'''

class State(rx.State):
    business_id: str = ""
    show_results: bool = False

    def submit(self):
        if self.business_id:
            self.show_results = True

    def go_back(self):
        self.show_results = False
        self.business_id = ""

    def handle_key_press(self, key: str):
        if key == "Enter":
            self.submit()

def header():
    return rx.box(
        rx.hstack(
            rx.heading("Business Review", color=colors["accent_green"], font_size="1.5em"),
            rx.spacer(),
            width="100%",
            padding="0 1em",
        ),
        width="100%",
        padding="0.5em",
        background=colors["accent_blue"],
    )

def home_page():
    return rx.center(
        rx.vstack(
            rx.text("What can I review for you today?", color=colors["text"], font_size="1.5em", margin_bottom="1em"),
            rx.input(
                placeholder="Business ID:",
                on_change=State.set_business_id,
                on_key_down=State.handle_key_press,
                value=State.business_id,
                width="100%",
                padding="0.5em",
                border_radius="0.3em",
                border=f"1px solid {colors['accent_blue']}",
                background=colors["input_bg"],
                color=colors["input_text"],
                _placeholder={"color": colors["text"], "opacity": 0.6},
            ),
            rx.button(
                "Review",
                on_click=State.submit,
                background=colors["accent_green"],
                color=colors["text"],
                padding="0.5em 2em",
                border_radius="0.3em",
                margin_top="1em",
                class_name="hover-button",
            ),
            rx.spacer(),
            rx.box(
                rx.html(reverted_scrolling_text),
                width="100%",
                height="50px",
                position="relative",
                overflow="hidden",
            ),
            spacing="1em",
            width="80%",
            max_width="600px",
            align_items="center",
            padding="2em",
            background="rgba(255, 255, 255, 0.8)",
            border_radius="1em",
            margin_top="40px",
            height="400px",
            position="relative",
        ),
        height="100vh",
        width="100%",
        background_image="url('/background.jpeg')",
        background_size="cover",
        background_position="center",
        background_repeat="no-repeat",
        background_attachment="fixed",
    )

def results_page():
    return rx.vstack(
        rx.text(f"Business: {State.business_id}", color=colors["accent_green"], font_size="1.5em", margin_bottom="1em"),
        rx.hstack(
            rx.box(
                rx.text("Results:", color=colors["accent_blue"], font_weight="bold", font_size="1.2em"),
                rx.text("Data goes here", color=colors["text"]),
                background=colors["input_bg"],
                padding="1.5em",
                border_radius="0.5em",
                width="48%",
                height="300px",
                border=f"1px solid {colors['accent_blue']}",
                overflow="auto",
            ),
            rx.box(
                rx.text("Next Steps:", color=colors["accent_blue"], font_weight="bold", font_size="1.2em"),
                rx.text("Steps go here", color=colors["text"]),
                background=colors["input_bg"],
                padding="1.5em",
                border_radius="0.5em",
                width="48%",
                height="300px",
                border=f"1px solid {colors['accent_blue']}",
                overflow="auto",
            ),
            width="100%",
            justify_content="space-between",
            margin_top="1em",
        ),
        rx.button(
            "Back",
            on_click=State.go_back,
            background=colors["accent_green"],
            color=colors["text"],
            padding="0.5em 2em",
            border_radius="0.3em",
            margin_top="2em",
            class_name="hover-button",
        ),
        width="90%",
        max_width="1000px",
        align_items="center",
        padding="2em",
        margin="0 auto",
    )

def index():
    return rx.box(
        rx.html(button_hover_effects),
        header(),
        rx.cond(
            State.show_results,
            results_page(),
            home_page(),
        ),
        width="100%",
        height="100vh",
        overflow="hidden",
    )

app = rx.App()
app.add_page(index)