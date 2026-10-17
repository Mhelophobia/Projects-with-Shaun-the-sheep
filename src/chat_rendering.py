"""Render chat tables without letting them widen the surrounding conversation."""

import re

import pandas as pd
import streamlit as st


def _cells(line):
    """Split Markdown table cells while preserving escaped pipes."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cell, code = [], [], None
    # Pipes in inline code are content (for example a pandas OR expression).
    for token in re.findall(r"\\.|`+|\||[^\\`|]+|\\", line):
        if token.startswith("`"):
            code = None if code == token else (token if code is None else code)
        if token == "|" and code is None:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append("|" if token == r"\|" else token)
    cells.append("".join(cell).strip())
    return cells


def message_blocks(message):
    """Yield Markdown or table blocks; leave fenced examples and malformed tables intact."""
    lines = message.splitlines()
    prose = []
    fence = None
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            prose.append(line)
            index += 1
            continue
        if fence is None and "|" in line and index + 1 < len(lines):
            headers = _cells(line)
            divider = _cells(lines[index + 1])
            if (len(headers) == len(divider) and
                    all(re.fullmatch(r":?-{3,}:?", cell) for cell in divider)):
                end = index + 2
                rows = []
                while end < len(lines) and "|" in lines[end] and lines[end].strip():
                    row = _cells(lines[end])
                    if len(row) != len(headers):
                        break
                    rows.append(row)
                    end += 1
                if rows:
                    if prose:
                        yield "markdown", "\n".join(prose)
                        prose = []
                    yield "table", (headers, rows)
                    index = end
                    continue
        prose.append(line)
        index += 1
    if prose:
        yield "markdown", "\n".join(prose)


_TABLE_READER = st.components.v2.component(
    "chat_table_reader",
    html="""
<button class="expand" type="button" aria-haspopup="dialog">↗ Expand table</button>
<dialog aria-label="Assistant table">
  <header><h2>Assistant table</h2><button class="close" type="button" aria-label="Close table and return to chat">×</button></header>
  <div class="table-scroll" tabindex="0" aria-label="Scrollable assistant table"><table><thead></thead><tbody></tbody></table></div>
</dialog>
""",
    css="""
button { font: inherit; cursor: pointer; color: var(--st-text-color); }
.expand { border: 1px solid var(--st-border-color, #dfe6ef); border-radius: 10px;
  padding: 8px 12px; background: var(--st-background-color, white); }
button:hover { color: var(--st-primary-color); border-color: var(--st-primary-color); }
button:focus-visible, .table-scroll:focus-visible { outline: 2px solid var(--st-primary-color); outline-offset: 2px; }
dialog { box-sizing: border-box; width: min(1120px, calc(100vw - 32px));
  max-width: calc(100vw - 32px); max-height: calc(100dvh - 40px); padding: 24px;
  border: 1px solid var(--st-border-color, #dfe6ef); border-radius: 20px;
  background: var(--st-background-color, white); color: var(--st-text-color, #162640);
  box-shadow: 0 16px 60px #14264033; font-family: inherit; }
dialog::backdrop { background: #14264055; }
header { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 20px; }
h2 { font-size: 1.4rem; margin: 0; }
.close { font-size: 28px; line-height: 1; background: transparent; border: 0; border-radius: 8px; width: 40px; height: 40px; }
.table-scroll { max-height: calc(100dvh - 170px); overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.6; }
th, td { text-align: left; vertical-align: top; padding: 12px; min-width: 110px;
  overflow-wrap: anywhere; border: 1px solid var(--st-border-color, #dfe6ef); }
th { background: var(--st-secondary-background-color, #f0f4fa); }
code { white-space: pre-wrap; font-size: .9em; }
""",
    js=r"""
export default function({ parentElement, data }) {
  const opener = parentElement.querySelector('.expand');
  const dialog = parentElement.querySelector('dialog');
  const close = parentElement.querySelector('.close');
  const hostWindow = dialog.ownerDocument.defaultView;
  // Construct text nodes rather than interpreting AI output as HTML.
  const fill = (cell, text) => {
    const parts = String(text).split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    for (const part of parts) {
      if (part.startsWith('**') && part.endsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = part.slice(2, -2); cell.append(strong);
      } else if (part.startsWith('`') && part.endsWith('`')) {
        const code = document.createElement('code');
        code.textContent = part.slice(1, -1); cell.append(code);
      } else { cell.append(document.createTextNode(part)); }
    }
  };
  const head = parentElement.querySelector('thead');
  const body = parentElement.querySelector('tbody');
  head.replaceChildren(); body.replaceChildren();
  const heading = document.createElement('tr');
  for (const label of data.headers) {
    const th = document.createElement('th'); th.scope = 'col'; fill(th, label); heading.append(th);
  }
  head.append(heading);
  for (const row of data.rows) {
    const tr = document.createElement('tr');
    for (const value of row) { const td = document.createElement('td'); fill(td, value); tr.append(td); }
    body.append(tr);
  }
  const dismiss = () => { dialog.close(); opener.focus({preventScroll: true}); };
  // Intercept Escape before the enclosing popover's document-level handler.
  if (dialog.escapeHandler) {
    hostWindow.removeEventListener('keydown', dialog.escapeHandler, true);
    hostWindow.removeEventListener('keyup', dialog.escapeHandler, true);
  }
  dialog.escapeHandler = event => {
    if (event.key === 'Escape' && (dialog.open || dialog.escapePending)) {
      event.preventDefault(); event.stopImmediatePropagation();
      dialog.escapePending = event.type === 'keydown';
      if (dialog.open) dismiss();
    }
  };
  hostWindow.addEventListener('keydown', dialog.escapeHandler, true);
  hostWindow.addEventListener('keyup', dialog.escapeHandler, true);
  opener.onclick = event => { event.stopPropagation(); dialog.showModal(); close.focus(); };
  close.onclick = event => { event.stopPropagation(); dismiss(); };
  dialog.oncancel = event => { event.preventDefault(); event.stopPropagation(); dismiss(); };
  dialog.onkeydown = event => {
    event.stopPropagation();
    if (event.key === 'Escape') { event.preventDefault(); dismiss(); }
  };
  dialog.onpointerdown = event => event.stopPropagation();
  dialog.onclick = event => {
    event.stopPropagation();
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom)) dismiss();
  };
  // No Streamlit state/trigger events: opening and closing never rerun Python.
  return () => {
    hostWindow.removeEventListener('keydown', dialog.escapeHandler, true);
    hostWindow.removeEventListener('keyup', dialog.escapeHandler, true);
    if (dialog.open) dialog.close();
  };
}
""",
)


def render_message(message, key="message"):
    for block_index, (kind, content) in enumerate(message_blocks(message)):
        if kind == "markdown":
            st.markdown(content)
        else:
            headers, rows = content
            # Stable internal names allow duplicate/empty display headings.
            columns = [f"column_{index}" for index in range(len(headers))]
            st.caption("Scroll across for more columns. Expand for easier reading.")
            st.dataframe(
                pd.DataFrame(rows, columns=columns),
                column_config={key: st.column_config.TextColumn(label, width=180)
                               for key, label in zip(columns, headers)},
                width="stretch", height=min(240, 36 * (len(rows) + 1) + 12),
                hide_index=True, alt="Table from the section assistant's reply",
            )
            _TABLE_READER(data={"headers": headers, "rows": rows},
                          key=f"{key}_table_{block_index}", height=46)
