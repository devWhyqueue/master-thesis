import type { Register } from 'claude-code'

const HEREDOC = /<<-?\s*['"\\]?\w/

export const register: Register = on => {
  on('tool.call', { tool: 'Bash' }, ($, e, next) => {
    const at = e.command.search(HEREDOC)
    return at >= 0 && e.command.slice(at).includes('\\\\')
      ? {
          deny:
            'heredoc-guard: this heredoc holds `\\\\`, which the Bash tool collapses to `\\` ' +
            'even under a quoted delimiter. Write the script to a file in the scratchpad ' +
            'and run that file.',
        }
      : next(e)
  })
}
