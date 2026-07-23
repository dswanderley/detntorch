# Security policy

## Supported version

Only the latest revision of the default branch is supported. This research
project does not maintain release branches or backport security fixes.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not disclose exploitable details in a public issue.

## Dependency policy

Dependabot checks Python dependencies weekly. Security updates should be
prioritized, and the `torch`/`torchvision` pair must be upgraded together.

Pillow must remain at version 12.3.0 or newer. Version 12.3.0 introduced the
fix for CVE-2026-54060, which adds decompression-bomb checks while compiling
bitmap font files.
