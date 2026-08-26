/** Conventional Commits, with the scopes this repo actually uses. */
export default {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "scope-enum": [
      2,
      "always",
      [
        "core",
        "models",
        "providers",
        "gateway",
        "web",
        "evals",
        "benchmarks",
        "infra",
        "docs",
        "spike",
      ],
    ],
    "body-max-line-length": [1, "always", 100],
  },
};
