class Color:
    @staticmethod
    def green(text: str) -> str:
        return f"\033[92m{text}\033[0m"