# Provenance

- Upstream: https://github.com/gcfy63821/dexx_lab , base commit `b7e4b6f42a6fde869e4218d53e17e29c9264a104`.
- Local knob implementation snapshot: `7913a8a` from that checkout (not pushed to upstream).
- This is a new standalone repository with new history, not an official upstream release.
- Included upstream components: robot assets/URDF/USD and their notices, camera calibration, deploy/actuation constants, shared Isaac robot configuration, the OakInk object-property helper required by that configuration, and asset build scripts.
- Added components: passive-knob assets and marks, CPU generation/render/export tools, experimental Isaac knob task, quality tests and portability instructions.
- Package initializers are replaced with inert initializers to avoid importing/registering upstream teacher/student environments and datasets. Shared configuration and robot geometry are otherwise preserved.
- Excluded: upstream policy implementations/checkpoints, demonstrations, dataset release exports, local logs and T2-z.
- Original MIT copyright/permission notice is retained in LICENSE. Robot model licenses and notices are preserved within assets/. Source files retain their existing license headers; see THIRD_PARTY_NOTICES.md.

CPU PyBullet is the verified dataset generator; Isaac and real-hardware behavior have not been validated by this extraction.
