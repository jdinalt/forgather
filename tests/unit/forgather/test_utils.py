"""
Unit tests for forgather.utils
"""

import pytest

from forgather.utils import (
    ADJECTIVES,
    SPECIES,
    AutoName,
    ConversionDescriptor,
    DiagnosticEnum,
    add_exception_notes,
    format_line_numbers,
    generate_name,
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
        # Species varies fastest within a fixed adjective.
        assert next(gen) == f"{ADJECTIVES[0]}_{SPECIES[0]}_"
        assert next(gen) == f"{ADJECTIVES[0]}_{SPECIES[1]}_"
        assert next(gen) == f"{ADJECTIVES[0]}_{SPECIES[2]}_"

    def test_sequence_unique(self):
        gen = iter(AutoName())
        names = [next(gen) for _ in range(50)]
        assert len(set(names)) == 50

    def test_adjective_rolls_over_after_all_species(self):
        gen = iter(AutoName())
        names = [next(gen) for _ in range(len(SPECIES) + 1)]
        # After exhausting every species for the first adjective, the second
        # adjective begins.
        assert names[len(SPECIES)] == f"{ADJECTIVES[1]}_{SPECIES[0]}_"

    def test_overflow_appends_numeric_suffix(self):
        gen = iter(AutoName())
        period = len(ADJECTIVES) * len(SPECIES)
        # Pull one full period plus one to force a wrap.
        names = [next(gen) for _ in range(period + 1)]
        assert names[0] == f"{ADJECTIVES[0]}_{SPECIES[0]}_"
        assert names[period] == f"{ADJECTIVES[0]}_{SPECIES[0]}_1_"

    def test_overflow_remains_unique(self):
        gen = iter(AutoName())
        period = len(ADJECTIVES) * len(SPECIES)
        names = [next(gen) for _ in range(period + 50)]
        assert len(set(names)) == len(names)

    def test_valid_python_identifiers(self):
        gen = iter(AutoName())
        names = [next(gen) for _ in range(10)]
        assert all(name.isidentifier() for name in names)

    def test_is_iterable(self):
        names = list(next(iter(AutoName())) for _ in range(3))
        assert len(names) == 3

    def test_new_iteration_restarts(self):
        auto = AutoName()
        first_run = [next(iter(auto)) for _ in range(3)]
        second_run = [next(iter(auto)) for _ in range(3)]
        assert first_run == second_run
        assert first_run[0] == f"{ADJECTIVES[0]}_{SPECIES[0]}_"


class TestGenerateName:
    def test_default_format(self):
        name = generate_name()
        adjective, sep, species = name.partition("-")
        assert sep == "-"
        assert adjective in ADJECTIVES
        assert species in SPECIES

    def test_custom_separator(self):
        name = generate_name(separator="_")
        adjective, sep, species = name.partition("_")
        assert adjective in ADJECTIVES
        assert species in SPECIES
        # An underscore separator yields a valid Python identifier.
        assert name.isidentifier()

    def test_varies(self):
        # Extremely unlikely to draw the same pair 50 times if random.
        names = {generate_name() for _ in range(50)}
        assert len(names) > 1


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
