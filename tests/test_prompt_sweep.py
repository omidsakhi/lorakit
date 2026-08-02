"""Tests for gendered prompt filtering and subject-trigger injection."""

from __future__ import annotations

import unittest

from lorakit.prompts import (
    Prompt,
    filter_prompts_by_gender,
    gender_class_words,
    inject_subject_trigger,
    normalize_gender,
)


class GenderFilterTests(unittest.TestCase):
    def test_normalize_gender_aliases(self):
        self.assertEqual(normalize_gender("Female"), "female")
        self.assertEqual(normalize_gender("woman"), "female")
        self.assertEqual(normalize_gender("M"), "male")
        with self.assertRaises(ValueError):
            normalize_gender("nonbinary")

    def test_gender_class_words(self):
        self.assertEqual(gender_class_words("female"), ("woman", "girl", "bride"))
        self.assertEqual(gender_class_words("male"), ("man", "boy", "groom"))

    def test_filter_female_includes_woman_girl_bride(self):
        prompts = [
            Prompt("1", "photo of woman, studio", "", 1),
            Prompt("2", "photo of girl, outdoor", "", 2),
            Prompt("3", "photo of bride, wedding", "", 3),
            Prompt("4", "photo of man, suit", "", 4),
            Prompt("5", "photo of groom, tuxedo", "", 5),
        ]
        ids = {p.id for p in filter_prompts_by_gender(prompts, "female")}
        self.assertEqual(ids, {"1", "2", "3"})

    def test_filter_male_includes_man_boy_groom(self):
        prompts = [
            Prompt("1", "photo of woman, studio", "", 1),
            Prompt("2", "photo of man, suit", "", 2),
            Prompt("3", "photo of boy, park", "", 3),
            Prompt("4", "photo of groom, tuxedo", "", 4),
        ]
        ids = {p.id for p in filter_prompts_by_gender(prompts, "male")}
        self.assertEqual(ids, {"2", "3", "4"})

    def test_man_does_not_match_inside_woman(self):
        prompts = [Prompt("1", "photo of woman, soft light", "", 1)]
        self.assertEqual(filter_prompts_by_gender(prompts, "male"), [])


class InjectSubjectTriggerTests(unittest.TestCase):
    def test_rewrites_woman(self):
        text = inject_subject_trigger(
            "portrait photo of woman, studio",
            trigger="sks",
            class_word="woman",
            gender="female",
        )
        self.assertEqual(text, "portrait photo of sks woman, studio")

    def test_rewrites_young_woman(self):
        text = inject_subject_trigger(
            "portrait photo of young woman, studio",
            trigger="sks",
            class_word="woman",
            gender="female",
        )
        self.assertEqual(text, "portrait photo of sks woman, studio")

    def test_rewrites_bride_and_girl_to_class_word(self):
        bride = inject_subject_trigger(
            "wedding photo of bride, veil",
            trigger="sks",
            class_word="woman",
            gender="female",
        )
        girl = inject_subject_trigger(
            "street photo of girl, rain",
            trigger="sks",
            class_word="woman",
            gender="female",
        )
        self.assertEqual(bride, "wedding photo of sks woman, veil")
        self.assertEqual(girl, "street photo of sks woman, rain")

    def test_rewrites_groom_for_male(self):
        text = inject_subject_trigger(
            "wedding photo of groom, tuxedo",
            trigger="sks",
            class_word="man",
            gender="male",
        )
        self.assertEqual(text, "wedding photo of sks man, tuxedo")

    def test_idempotent_when_trigger_present(self):
        text = "portrait photo of sks woman, studio"
        self.assertEqual(
            inject_subject_trigger(text, trigger="sks", class_word="woman", gender="female"),
            text,
        )


if __name__ == "__main__":
    unittest.main()
