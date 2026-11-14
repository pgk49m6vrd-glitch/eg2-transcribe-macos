# Contributing guidelines

Thanks for helping improve this project. A healthy contribution flow makes review easier, lowers the chance of regressions, and keeps the project welcoming for everyone.

This project values clear communication, focused changes, and respectful collaboration. The following practices are expected for pull requests, issues, and feedback.

## 1. Before opening an issue

Open an issue only when you have a concrete bug, feature request, or improvement idea that can be discussed clearly.

Please do:
- Check whether the issue already exists before creating a duplicate.
- Use a clear, specific title.
- Include the context that matters: what happened, what you expected, and what version/environment you were using.
- Share reproduction steps, logs, screenshots, or sample files when useful.
- Keep the report focused on one problem at a time.

Please avoid:
- Posting vague reports like "it does not work" without reproduction details.
- Mixing multiple unrelated bugs into one issue.
- Opening issues to ask for support that belongs in a PR discussion or direct feedback channel.

A good issue usually includes:
- Summary of the problem
- Steps to reproduce
- Expected result
- Actual result
- Relevant environment details

Example:

```md
Title: Audio transcription hangs when dragging a .wav file into the Studio window

Description:
- macOS version: 15.0
- App version: 1.2.3
- File type: .wav, 24-bit, 16 kHz

Steps:
1. Open the Studio window
2. Drag a file into the drop zone
3. Start transcription

Expected: the file is transcribed normally.
Actual: the app freezes and stops responding.
```

## 2. Before opening a pull request

Pull requests should be small, reviewable, and directly related to a single goal.

Please do:
- Keep the scope narrow and understandable.
- Explain what problem the change solves.
- Mention any trade-offs, edge cases, or follow-up work.
- Run the relevant checks or validations before asking for review.
- Ensure the branch is in a clean, reviewable state.

Please avoid:
- Bundling unrelated cleanup with functional changes.
- Reformatting unrelated files.
- Making large refactors without a clear reason.
- Submitting work that has not been checked locally.

A strong PR description usually includes:
- Why the change is needed
- What was changed
- Any validation performed
- Notes on risk or limitations

Example:

```md
## Summary
This PR improves the app's file-drop handling so unsupported audio formats fail gracefully instead of freezing the Studio window.

## Why
Users were able to drop files that the app could not process. The UI stayed stuck in a loading state and required a restart.

## Changes
- Added validation for unsupported media formats before starting transcription.
- Show a clear error message instead of leaving the app in a frozen state.
- Added a focused regression test for the invalid-file case.

## Validation
- Ran the relevant unit tests
- Manually tested drag-and-drop with a .wav file and an unsupported extension
```

## 3. Review and communication expectations

- Be respectful and constructive in reviews.
- Focus on the code and the problem, not the author.
- Prefer actionable feedback with clear reasoning.
- Ask questions when something is unclear, instead of assuming intent.
- Keep timelines realistic; not all feedback is immediate.

Good review feedback is specific and helpful:

```md
This looks close, but the error path still leaves the UI in a loading state. Could we move the status reset into a `finally` block so it always clears, even when the transcription throws?
```

Less helpful feedback:

```md
This is wrong.
```

## 4. Issue and PR etiquette

- Use clear titles and plain language.
- Respect the maintainers' time by keeping discussions on-topic.
- Do not demand fast review without context.
- If you are unsure whether a topic belongs in an issue or a PR, open the smallest relevant discussion first.
- If a change is incomplete, say so clearly rather than pretending it is ready.

## 5. Feedback by email

For private feedback, general suggestions, or non-public concerns, you can contact the maintainer at:

```text
atlascorejsx@proton.me
```

Use email when:
- The topic is sensitive or personal
- You prefer not to open a public issue
- You want to share feedback privately

When sending feedback by email, include:
- A short description of the issue or suggestion
- The relevant project context
- Any reproduction details or references
- The best way to follow up with you

## 6. Contribution checklist

Before submitting anything, check:
- [ ] The change is scoped and understandable.
- [ ] The issue or feature request is clearly described.
- [ ] The PR/issue title is specific.
- [ ] The work has been checked locally.
- [ ] The explanation is clear enough for reviewers.
- [ ] The communication is respectful and constructive.

## 7. Final note

Good contributions are not only about code quality. They are also about clarity, accountability, and making the project easier for others to maintain and extend. Thank you for helping keep this project useful, reliable, and welcoming.
