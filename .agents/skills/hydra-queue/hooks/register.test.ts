import { expect, mock, test } from 'claude-code/testing'
import type { On } from 'claude-code'

const QUEUED = '5031922,0,PENDING,imb-fit\n5031922,1,PENDING,imb-fit\n5031920,28,RUNNING,imb-fit\n---\n'

function world(on: On, stdout: string): { ran: string[] } {
  const ran: string[] = []
  mock.clock(on)
  on('process.run', () => ({
    value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false },
  }))
  on('tool.call', { tool: 'Bash' }, ($, e) => {
    ran.push(e.command)
    return { result: { stdout: '', stderr: '', interrupted: false } }
  })
  return { ran }
}

test('denies an explicit --time', async ($, on) => {
  const { ran } = world(on, '---\n')
  const cli = await $.tool.call({ tool: 'Bash', command: 'ssh hydra "sbatch --time=01:00:00 job.sh"' })
  const script = await $.tool.call({ tool: 'Bash', command: 'echo "#SBATCH --time=02:00:00" > job.sh' })
  expect(cli.deny).toContain('--time')
  expect(script.deny).toContain('--time')
  expect(ran).toEqual([])
})

test('denies a dependency on a job that left the queue', async ($, on) => {
  const { ran } = world(on, QUEUED)
  const stale = await $.tool.call({
    tool: 'Bash',
    command: 'ssh hydra "sbatch --dependency=afterok:4000000 job.sh"',
  })
  expect(stale.deny).toContain('4000000')
  expect(ran).toEqual([])
})

test('shows the queue once, then lets the identical submit through', async ($, on) => {
  const { ran } = world(on, QUEUED)
  const command = 'ssh hydra "python -m imbalance_benchmark submit --stage fit"'
  const first = await $.tool.call({ tool: 'Bash', command })
  expect(first.deny).toContain('5031922 imb-fit PENDING [0-1]')
  expect(first.deny).toContain('5031920 imb-fit RUNNING [28]')
  const second = await $.tool.call({ tool: 'Bash', command })
  expect(second.deny).toBeUndefined()
  const third = await $.tool.call({ tool: 'Bash', command })
  expect(third.deny).toContain('5031922')
  expect(ran).toEqual([command])
})

test('leaves other commands and an empty queue alone', async ($, on) => {
  const { ran } = world(on, '---\n')
  await $.tool.call({ tool: 'Bash', command: 'git status' })
  await $.tool.call({ tool: 'Bash', command: 'ssh hydra "sbatch job.sh"' })
  expect(ran.length).toBe(2)
})
