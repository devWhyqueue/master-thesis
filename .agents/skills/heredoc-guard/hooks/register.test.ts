import { expect, test } from 'claude-code/testing'

test('refuses a heredoc with doubled backslashes only', async ($, on) => {
  const ran: string[] = []
  on('tool.call', { tool: 'Bash' }, ($, e) => {
    ran.push(e.command)
    return { result: { stdout: '', stderr: '', interrupted: false } }
  })

  const doubled = "python - <<'PY'\nprint('Section~\\\\ref{a}')\nPY"
  const single = "python - <<'PY'\nprint('a\\nb')\nPY"
  const noHeredoc = "grep 'a\\\\b' file.tex"

  expect((await $.tool.call({ tool: 'Bash', command: doubled })).deny).toContain('heredoc-guard')
  await $.tool.call({ tool: 'Bash', command: single })
  await $.tool.call({ tool: 'Bash', command: noHeredoc })
  expect(ran).toEqual([single, noHeredoc])
})
