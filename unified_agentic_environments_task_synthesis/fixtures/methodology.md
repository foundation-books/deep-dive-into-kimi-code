# Analysis audit methodology

For each fictional classifier, report the arithmetic mean of its two class
recalls. A class recall is correct predictions divided by that class's support.
Also report ordinary normalized accuracy, the sum of correct predictions divided
by total support. The required ranking uses balanced accuracy. The public
definitions are identified in `metric_definitions.md`.

The input is single-label, unweighted, and uses `adjusted=False`. Each
classifier must have exactly two distinct class labels. Both supports must be
positive integers. Correct counts must be integers between zero and support.
Negative counts, excessive correct counts, duplicate or missing classes, and
zero support are invalid inputs.
