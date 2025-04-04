from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout


class CareerQuizApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.question_label = Label(text="What interests you the most?", font_size=18)
        layout.add_widget(self.question_label)

        # Scrollable options
        scroll_view = ScrollView()
        self.options_layout = GridLayout(cols=1, spacing=5, size_hint_y=None)
        self.options_layout.bind(minimum_height=self.options_layout.setter('height'))

        options = [
            "Technology & Coding",
            "Science & Research",
            "Business & Entrepreneurship",
            "Arts & Creativity",
            "Social Work & Teaching"
        ]

        for option in options:
            btn = Button(text=option, size_hint_y=None, height=50)
            btn.bind(on_press=self.select_career)
            self.options_layout.add_widget(btn)

        scroll_view.add_widget(self.options_layout)
        layout.add_widget(scroll_view)

        self.result_label = Label(text="", font_size=16)
        layout.add_widget(self.result_label)

        return layout

    def select_career(self, instance):
        career_paths = {
            "Technology & Coding": "You should explore careers in Software Engineering, AI, or Data Science!",
            "Science & Research": "Consider careers in Medicine, Biotechnology, or Physics Research!",
            "Business & Entrepreneurship": "You might thrive as an Entrepreneur, Manager, or Business Analyst!",
            "Arts & Creativity": "Design, Writing, and Film-making could be great career paths for you!",
            "Social Work & Teaching": "You could make a great Teacher, Counselor, or Social Worker!"
        }
        self.result_label.text = career_paths[instance.text]


if __name__ == "__main__":
    CareerQuizApp().run()
