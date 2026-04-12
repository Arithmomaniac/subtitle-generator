/**
 * Unit tests for subtitle-vm.js — zero-framework pattern.
 * Run with: node web/tests/test-subtitle-vm.js
 */

import { deriveSubtitleVM, deriveSourcesVM, buildSettingsVM } from "../js/subtitle-vm.js";

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) { passed++; }
  else { failed++; console.error("  FAIL:", msg); }
}

function test(name, fn) {
  try { fn(); }
  catch (e) { failed++; console.error("  FAIL:", name, e.message); }
}

// ── deriveSubtitleVM ──

test("null input returns empty", () => {
  const vm = deriveSubtitleVM(null);
  assert(vm.slots.length === 0, "slots should be empty");
  assert(vm.fullText === "", "fullText should be empty");
  assert(vm.remixed === false, "remixed should be false");
});

test("error response returns empty", () => {
  const vm = deriveSubtitleVM({ error: "bad request" });
  assert(vm.slots.length === 0, "slots should be empty on error");
});

test("basic non-remixed subtitle", () => {
  const sub = {
    text: "A, B, and the C of D",
    item1: "A", item2: "B", action_noun: "C", of_object: "D",
    remixed: false, remix_parts: {},
  };
  const vm = deriveSubtitleVM(sub);
  assert(vm.fullText === "A, B, and the C of D", "fullText");
  assert(vm.remixed === false, "not remixed");
  assert(vm.similarity === null, "no similarity");
  // Slots: A, comma, B, "and the", C, "of", D = 7
  assert(vm.slots.length === 7, `expected 7 slots, got ${vm.slots.length}`);
  assert(vm.slots[0].text === "A", "slot 0 text");
  assert(vm.slots[0].cls === "slot-list1", "slot 0 class");
  assert(vm.slots[1].isPunc === true, "slot 1 is punctuation");
  assert(vm.slots[6].text === "D", "slot 6 text");
  assert(vm.slots[6].cls === "slot-of", "slot 6 class");
});

test("remixed compound (modifier + head)", () => {
  const sub = {
    text: "A, B, and the C of American Democracy",
    item1: "A", item2: "B", action_noun: "C", of_object: "American Democracy",
    remixed: true, remix_parts: { modifier: "American", head: "Democracy" },
    remix_similarity: 0.72,
  };
  const vm = deriveSubtitleVM(sub);
  assert(vm.remixed === true, "should be remixed");
  assert(vm.similarity === 0.72, "similarity should be 0.72");
  // Slots: A, comma, B, "and the", C, "of", American, Democracy = 8
  assert(vm.slots.length === 8, `expected 8 slots, got ${vm.slots.length}`);
  assert(vm.slots[6].text === "American", "modifier slot");
  assert(vm.slots[6].cls === "slot-subpart", "modifier class");
  assert(vm.slots[7].text === "Democracy", "head slot");
});

test("remixed prepositional (topic + prep + complement)", () => {
  const sub = {
    text: "A, B, and the C of Jews in America",
    item1: "A", item2: "B", action_noun: "C", of_object: "Jews in America",
    remixed: true, remix_parts: { topic: "Jews", prep: "in", complement: "America" },
  };
  const vm = deriveSubtitleVM(sub);
  // Slots: A, comma, B, "and the", C, "of", Jews, "in", America = 9
  assert(vm.slots.length === 9, `expected 9 slots, got ${vm.slots.length}`);
  assert(vm.slots[6].text === "Jews", "topic slot");
  assert(vm.slots[7].isPunc === true, "prep is punctuation");
  assert(vm.slots[7].text === "in", "prep text");
  assert(vm.slots[8].text === "America", "complement slot");
});

// ── deriveSourcesVM ──

test("null input returns empty array", () => {
  const sources = deriveSourcesVM(null);
  assert(Array.isArray(sources) && sources.length === 0, "should be empty array");
});

test("basic sources", () => {
  const sub = {
    item1: "Jefferson", item2: "Cats", action_noun: "history", of_object: "desire",
    sources: {
      item1: { title: "Thomas Jefferson and...", tag: "LOC" },
      item2: { title: null, tag: null },
      action_noun: { title: "A History of...", tag: "OL" },
      of_object: { title: "The Geography of Desire", tag: "LOC" },
    },
  };
  const sources = deriveSourcesVM(sub);
  assert(sources.length === 4, `expected 4 sources, got ${sources.length}`);
  assert(sources[0].label === "List item 1", "first label");
  assert(sources[0].filler === "Jefferson", "first filler");
  assert(sources[0].book === "Thomas Jefferson and...", "first book");
  assert(sources[1].book === null, "null book for missing source");
});

test("remix sources resolve parts", () => {
  const sub = {
    item1: "A", item2: "B", action_noun: "C", of_object: "American Democracy",
    remix_parts: { modifier: "American", head: "Democracy" },
    sources: {
      item1: { title: "t1", tag: "LOC" },
      item2: { title: "t2", tag: "LOC" },
      action_noun: { title: "t3", tag: "OL" },
      of_modifier: { title: "Source of American", tag: "LOC" },
      of_head: { title: "Source of Democracy", tag: "OL" },
    },
  };
  const sources = deriveSourcesVM(sub);
  assert(sources.length === 5, `expected 5 sources, got ${sources.length}`);
  const modSrc = sources.find(s => s.label === "Modifier");
  assert(modSrc && modSrc.filler === "American", "modifier filler resolved");
  assert(modSrc && modSrc.book === "Source of American", "modifier book");
});

// ── buildSettingsVM ──

test("local mode shows model", () => {
  const vm = buildSettingsVM("local");
  assert(vm.showModel === true, "showModel");
});

test("azure mode hides model", () => {
  const vm = buildSettingsVM("azure");
  assert(vm.showModel === false, "hideModel");
});

// ── Summary ──
console.log(`\n${passed} passed, ${failed} failed (${passed + failed} total)`);
process.exit(failed > 0 ? 1 : 0);
