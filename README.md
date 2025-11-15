diff --git a/README.md b/README.md
index e3094f3dab97da3569db000e8ea5da994c61f314..4eb445c02cf19dcd825b792f0b42600b120442dd 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,41 @@
 # TITLE
 Short purpose, quickstart, minimal example.
+
+## Cloning the repository locally
+
+If you have not yet created a working copy on your machine, clone the
+GitHub repository into the directory where you keep your projects:
+
+```bash
+cd /path/where/you/store/projects
+git clone git@github.com:<your-org>/qte-attestation.git
+cd qte-attestation
+```
+
+Replace `<your-org>` with the actual owner of the remote repository. If you
+prefer HTTPS, swap the clone URL accordingly, for example:
+
+```bash
+git clone https://github.com/<your-org>/qte-attestation.git
+```
+
+Once the repository is cloned, you can pull updates later with
+`git fetch origin` followed by `git pull` (or `git reset --hard origin/main`
+if you need to discard local changes and exactly match the remote `main`
+branch).
+
+## Running the test suite
+
+All project regression tests live in the `tests/` directory at the
+repository root. To execute them from a fresh checkout:
+
+```bash
+cd qte-attestation
+pytest
+```
+
+`pytest` will automatically discover both the provenance hashing coverage
+and the Unit 01 bundle invariants. If the `tests/` directory is missing in
+your local clone, make sure you have pulled the latest commits from the
+`work` branch (or whichever branch contains the most recent work) and that
+no sparse checkout rules are hiding repository paths.

