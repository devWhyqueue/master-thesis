import type { EngineInterface, Register } from 'claude-code'

const POLL_MS = 180_000
const ACK_MS = 600_000
const SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'hydra']
// Comma-separated: a `|` would be a pipe inside the remote `bash -lc`.
const QUEUE =
  "bash -lc 'squeue --me -h -r -o %F,%K,%T,%j; echo ---; " +
  "sacct -X -n -P -S now-1hour -s F,TO,OOM,NF -o JobID,JobName,State'"

const SCRIPT_TIME = /#SBATCH\s+(--time|-t)[=\s]\s*\d/
const CLI_TIME = /\bsbatch\b[^|;&\n]*\s(--time|-t)[=\s]/
const SUBMIT = /\bssh\b[\s\S]*\b(sbatch|submit)\b/
const DEPENDENCY = /\bafter\w*:([\d_:]+)/g

const NO_TIME =
  'hydra-queue: no explicit --time. Each Hydra partition caps its own time; ' +
  'a tighter limit kills running tasks. Pick a longer partition instead.'

type Task = { job: string; index: string; state: string; name: string }
type Queue = { tasks: Task[]; failed: string[] }

async function readQueue($: EngineInterface): Promise<Queue | undefined> {
  try {
    const ran = await $.process.run([...SSH, QUEUE], { timeoutMs: 30_000 })
    if (ran.exitCode !== 0) return undefined
    const [live = '', ended = ''] = ran.stdout.split('---')
    const lines = (text: string) => text.split('\n').map(line => line.trim()).filter(Boolean)
    return {
      tasks: lines(live).map(line => {
        const [job = '', index = '', state = '', ...name] = line.split(',')
        return { job, index, state, name: name.join(',') }
      }),
      failed: lines(ended).map(line => {
        const [id, name, state] = line.split('|')
        return `${name} ${id} ${state}`
      }),
    }
  } catch {
    return undefined
  }
}

function ranges(indices: number[]): string {
  const sorted = [...indices].sort((a, b) => a - b)
  const parts: string[] = []
  for (let i = 0; i < sorted.length; i++) {
    const start = sorted[i]
    while (sorted[i + 1] === sorted[i]! + 1) i++
    parts.push(start === sorted[i] ? `${start}` : `${start}-${sorted[i]}`)
  }
  return parts.join(',')
}

function summary(tasks: Task[]): string {
  const groups = new Map<string, number[]>()
  for (const task of tasks) {
    const key = `${task.job} ${task.name} ${task.state}`
    const indices = groups.get(key) ?? []
    if (/^\d+$/.test(task.index)) indices.push(Number(task.index))
    groups.set(key, indices)
  }
  return [...groups]
    .map(([key, indices]) => (indices.length ? `${key} [${ranges(indices)}]` : key))
    .join('\n')
}

// Commands denied once with the queue shown; the identical retry passes.
const acknowledged = new Map<string, number>()

async function refusal($: EngineInterface, command: string): Promise<string | undefined> {
  if (SCRIPT_TIME.test(command) || CLI_TIME.test(command)) return NO_TIME
  if (!SUBMIT.test(command) || command.includes('--dry-run')) return undefined

  const queue = await readQueue($)
  if (queue === undefined)
    return 'hydra-queue: could not read squeue over `ssh hydra`. Fix SSH before submitting.'

  const live = new Set(queue.tasks.map(task => task.job))
  const stale = [...command.matchAll(DEPENDENCY)]
    .flatMap(match => match[1]!.split(':'))
    .map(id => id.split('_')[0]!)
    .filter(id => id !== '' && !live.has(id))
  if (stale.length > 0)
    return (
      `hydra-queue: dependency on job ${stale.join(', ')}, which is not in squeue. ` +
      'A finished job ages out and sbatch then fails with "Job dependency problem". ' +
      'Verify its output on disk, then submit without --dependency.'
    )

  if (queue.tasks.length === 0) return undefined
  if (Date.now() - (acknowledged.get(command) ?? 0) < ACK_MS) {
    acknowledged.delete(command)
    return undefined
  }
  acknowledged.set(command, Date.now())
  return (
    'hydra-queue: jobs already queued or running (job name state [array indices]):\n' +
    `${summary(queue.tasks)}\n` +
    'A "remaining work" list comes from disk state only and includes in-flight tasks. ' +
    'Diff the exact indices this submit creates against the list above. ' +
    'Re-run the identical command to proceed.'
  )
}

const SHORT: Record<string, string> = { RUNNING: 'R', PENDING: 'PD' }
let seenFailed: Set<string> | undefined

async function poll($: EngineInterface): Promise<void> {
  const queue = await readQueue($)
  if (queue === undefined) return $.ui.status('hydra ?')

  const counts = new Map<string, number>()
  for (const task of queue.tasks) {
    const state = SHORT[task.state] ?? task.state
    counts.set(state, (counts.get(state) ?? 0) + 1)
  }
  const parts = [...counts].map(([state, count]) => `${state} ${count}`)
  $.ui.status(parts.length > 0 ? `hydra ${parts.join(' · ')}` : undefined)

  // The first poll only seeds, so a reload does not repeat old failures.
  const fresh = queue.failed.filter(line => seenFailed !== undefined && !seenFailed.has(line))
  if (fresh.length > 0) {
    const more = fresh.length > 3 ? ` +${fresh.length - 3} more` : ''
    $.ui.toast(`hydra: ${fresh.slice(0, 3).join(', ')}${more}`, { timeoutMs: 15_000 })
  }
  seenFailed = new Set([...(seenFailed ?? []), ...queue.failed])
}

export const register: Register = on => {
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const deny = await refusal($, e.command)
    return deny === undefined ? next(e) : { deny }
  })
  on('tool.call', { tool: 'PowerShell' }, async ($, e, next) => {
    const deny = await refusal($, e.command)
    return deny === undefined ? next(e) : { deny }
  })
  on('tool.call', { tool: 'Write' }, ($, e, next) =>
    SCRIPT_TIME.test(e.content) ? { deny: NO_TIME } : next(e),
  )
  on('tool.call', { tool: 'Edit' }, ($, e, next) =>
    SCRIPT_TIME.test(e.new_string) ? { deny: NO_TIME } : next(e),
  )

  on('session.start', ($, e, next) => {
    void poll($)
    $.clock.every(POLL_MS, () => void poll($))
    return next(e)
  })
}
