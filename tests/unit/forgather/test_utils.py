"""
Unit tests for forgather.utils
"""

import pytest

from forgather.utils import (
    AutoName,
    ConversionDescriptor,
    DiagnosticEnum,
    add_exception_notes,
    format_line_numbers,
    indent_block,
    track_depth,
)


class TestFormatLineNumbers:
    def test_single_line(self):
        result = format_line_numbers("hello")
        assert "1" in result
        assert "hello" in result

    def test_multiple_lines(self):
        result = format_line_numbers("line1\nline2\nline3")
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "line1" in result
        assert "line3" in result

    def test_empty_string(self):
        result = format_line_numbers("")
        assert "1" in result

    def test_line_numbers_right_aligned(self):
        # Numbers should be right-aligned to 6 chars
        result = format_line_numbers("a")
        assert result.startswith("     1: ")


class TestAddExceptionNotes:
    def test_returns_same_exception(self):
        exc = Exception("original message")
        result = add_exception_notes(exc, "extra context")
        assert result is exc

    def test_note_appears_in_notes(self):
        exc = Exception("original message")
        add_exception_notes(exc, "extra context")
        assert "extra context" in exc.__notes__

    def test_original_message_preserved(self):
        exc = Exception("keep this")
        add_exception_notes(exc, "note")
        assert "keep this" in exc.args[0]

    def test_multiple_notes(self):
        exc = Exception("base")
        add_exception_notes(exc, "note1", "note2")
        assert "note1" in exc.__notes__
        assert "note2" in exc.__notes__

    def test_notes_are_coerced_to_str(self):
        exc = Exception("msg")
        add_exception_notes(exc, 42)
        assert "42" in exc.__notes__

    def test_exception_with_no_string_arg_works(self):
        """Non-string exception args are no longer a problem."""
        exc = Exception(42, 99)
        result = add_exception_notes(exc, "some note")
        assert result is exc
        assert "some note" in exc.__notes__

    def test_notes_accumulate_across_calls(self):
        exc = Exception("msg")
        add_exception_notes(exc, "first")
        add_exception_notes(exc, "second")
        assert "first" in exc.__notes__
        assert "second" in exc.__notes__


class TestAutoName:
    def test_first_names(self):
        gen = iter(AutoName())
        assert next(gen) == "alpha_"
        assert next(gen) == "beta_"
        assert next(gen) == "gamma_"

    def test_last_single_name(self):
        gen = iter(AutoName())
        names = [next(gen) for _ in range(24)]
        assert names[23] == "omega_"

    def test_wraps_to_two_letter_names(self):
        gen = iter(AutoName())
        # Skip first 24 single-letter names (alpha_ through omega_)
        for _ in range(24):
            next(gen)
        # The 25th name: i=24, NAMES[24%24]="alpha" then NAMES[(24//24)%24]="beta"
        # The least-significant component is the rightmost → "beta_alpha_"
        assert next(gen) == "beta_alpha_"

    def test_second_two_letter_name(self):
        gen = iter(AutoName())
        for _ in range(25):
            next(gen)
        # i=25: NAMES[25%24]="beta" then NAMES[(25//24)%24]="beta" → "beta_beta_"
        assert next(gen) == "beta_beta_"

    def test_is_iterable(self):
        names = list(next(iter(AutoName())) for _ in range(3))
        assert len(names) == 3

    def test_new_iteration_restarts(self):
        auto = AutoName()
        first_run = [next(iter(auto)) for _ in range(3)]
        second_run = [next(iter(auto)) for _ in range(3)]
        assert first_run == second_run


class TestTrackDepth:
    def test_depth_increments_during_call(self):
        captured = []

        class MyClass:
            level = 0

            @track_depth
            def method(self):
                captured.append(self.level)

        obj = MyClass()
        obj.method()
        assert captured == [1]  # level was 1 inside the method

    def test_depth_restored_after_call(self):
        class MyClass:
            level = 0

            @track_depth
            def method(self):
                pass

        obj = MyClass()
        obj.method()
        assert obj.level == 0

    def test_depth_restored_on_exception(self):
        class MyClass:
            level = 0

            @track_depth
            def method(self):
                raise ValueError("test error")

        obj = MyClass()
        with pytest.raises(ValueError):
            obj.method()
        assert obj.level == 0

    def test_nested_calls_increment(self):
        captured = []

        class MyClass:
            level = 0

            @track_depth
            def outer(self):
                captured.append(("outer", self.level))
                self.inner()

            @track_depth
            def inner(self):
                captured.append(("inner", self.level))

        obj = MyClass()
        obj.outer()
        assert captured == [("outer", 1), ("inner", 2)]
        assert obj.level == 0


class TestConversionDescriptor:
    def test_basic_conversion(self):
        class MyClass:
            value = ConversionDescriptor(int, default=0)

        obj = MyClass()
        obj.value = "42"
        assert obj.value == 42
        assert isinstance(obj.value, int)

    def test_default_value(self):
        class MyClass:
            value = ConversionDescriptor(int, default=99)

        obj = MyClass()
        assert obj.value == 99

    def test_class_access_returns_default(self):
        class MyClass:
            value = ConversionDescriptor(int, default=0)

        # Accessing on the class (not instance) returns the default
        assert MyClass.value == 0

    def test_float_conversion(self):
        class MyClass:
            ratio = ConversionDescriptor(float, default=0.0)

        obj = MyClass()
        obj.ratio = "3.14"
        assert abs(float(obj.ratio) - 3.14) < 1e-10


class TestDiagnosticEnum:
    def setup_method(self):
        class Color(DiagnosticEnum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        self.Color = Color

    def test_valid_value(self):
        assert self.Color("red") == self.Color.RED

    def test_invalid_value_raises_with_helpful_message(self):
        with pytest.raises(ValueError, match="choose one of"):
            self.Color("purple")

    def test_error_message_includes_valid_choices(self):
        with pytest.raises(ValueError) as exc_info:
            self.Color("invalid")
        error_msg = str(exc_info.value)
        assert "red" in error_msg or "green" in error_msg or "blue" in error_msg


class TestIndentBlock:
    def test_indent_block_default_indent(self):
        """
        indent_block() should use a default indent of 4 spaces.
        """
        result = indent_block("some text")
        assert result.startswith("    ")
