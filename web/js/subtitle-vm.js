/**
 * Pure view-model derivation functions for subtitle-generator.
 * No DOM, no side effects, no framework — just data transformation.
 */

/** Slot CSS class by key prefix. */
const SLOT_CLASSES = {
  item1: "slot-list1",
  item2: "slot-list2",
  action_noun: "slot-action",
  of_object: "slot-of",
  of_modifier: "slot-subpart",
  of_head: "slot-subpart",
  of_topic: "slot-subpart",
  of_complement: "slot-subpart",
};

/**
 * Derive display-ready slot data from an API generate response.
 * @param {object} sub - raw API response from /api/generate
 * @returns {{ slots: Array<{text, cls, isPunc}>, fullText: string, remixed: boolean, similarity: number|null }}
 */
export function deriveSubtitleVM(sub) {
  if (!sub || sub.error) {
    return { slots: [], fullText: "", remixed: false, similarity: null };
  }

  const actionArt = (sub.action_article || "the");
  const slots = [
    { text: sub.item1, cls: SLOT_CLASSES.item1 },
    { text: ",", cls: "", isPunc: true },
    { text: sub.item2, cls: SLOT_CLASSES.item2 },
    { text: `, and ${actionArt}`, cls: "", isPunc: true },
    { text: sub.action_noun, cls: SLOT_CLASSES.action_noun },
    { text: sub.of_article ? `of ${sub.of_article}` : "of", cls: "", isPunc: true },
  ];

  if (sub.remixed && sub.remix_parts) {
    if (sub.remix_parts.modifier !== undefined) {
      slots.push({ text: sub.remix_parts.modifier, cls: SLOT_CLASSES.of_modifier });
      slots.push({ text: sub.remix_parts.head, cls: SLOT_CLASSES.of_head });
    } else if (sub.remix_parts.topic !== undefined) {
      slots.push({ text: sub.remix_parts.topic, cls: SLOT_CLASSES.of_topic });
      slots.push({ text: sub.remix_parts.prep, cls: "", isPunc: true });
      slots.push({ text: sub.remix_parts.complement, cls: SLOT_CLASSES.of_complement });
    }
  } else {
    slots.push({ text: sub.of_object, cls: SLOT_CLASSES.of_object });
  }

  return {
    slots,
    fullText: sub.text || "",
    remixed: !!sub.remixed,
    similarity: sub.remix_similarity ?? null,
  };
}

/** Source display labels by API key. */
const SOURCE_LABELS = {
  item1: "List item 1",
  item2: "List item 2",
  action_noun: "Action noun",
  of_object: "Of-object",
  of_modifier: "Modifier",
  of_head: "Head",
  of_topic: "Topic",
  of_complement: "Complement",
};

/**
 * Derive source display data from an API generate response.
 * @param {object} sub - raw API response from /api/generate
 * @returns {Array<{label, filler, book, tag}>}
 */
export function deriveSourcesVM(sub) {
  if (!sub || !sub.sources) return [];

  return Object.entries(sub.sources).map(([key, src]) => {
    const label = SOURCE_LABELS[key] || key;
    // Resolve the filler text from the right place
    let filler = sub[key];
    if (!filler && sub.remix_parts) {
      const partKey = key.replace("of_", "");
      filler = sub.remix_parts[partKey];
    }
    return {
      label,
      filler: filler || "",
      book: src.title || null,
      tag: src.tag || null,
    };
  });
}

/**
 * Determine which settings are visible based on mode.
 * @param {"local"|"azure"|string} mode
 * @returns {{ showRemixProb, showMinSim, showModel, showDeepResearch }}
 */
export function buildSettingsVM(mode) {
  const isLocal = mode === "local";
  return {
    showModel: isLocal,
  };
}

/**
 * Clean jacket markdown: keep only the template sections, strip preamble/postamble.
 * @param {string} md - raw markdown from LLM
 * @returns {string} cleaned markdown
 */
export function cleanJacketMarkdown(md) {
  if (!md) return "";
  // Find first ## heading
  const firstH2 = md.search(/^## /m);
  if (firstH2 > 0) md = md.slice(firstH2);
  // Remove trailing content after the last section's content
  // (anything after a line that's clearly not part of the jacket)
  return md.trimEnd();
}
