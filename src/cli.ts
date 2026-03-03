#!/usr/bin/env node

import { Command } from 'commander';
import { AgentsRunner, type AgentsRunnerContext } from './agents/agents-runner.js';
import { randomUUID } from 'node:crypto';

if (process.env.DEBUG === 'false') {
  console.debug = () => {};
}

const program = new Command();

program
  .name('adeline')
  .description('Run a multi agent setup with orchestrator agent')
  .version('0.0.1');

program
  .command('run')
  .description('Run a multi agent setup with orchestrator agent')
  .option('-f, --file <file_name>', 'context file')
  .action(async (options) => {
    const runner = new AgentsRunner();

    try {
      await runner.initialize();

      const executionId = `exec_${Date.now()}_${randomUUID().split('-')[0]}`;

      const context: AgentsRunnerContext = {
        executionId,
        filePath: options.file,
      };

      const result = await runner.runJob(context);

      if (result.status === 'completed') {
        process.stdout.write(typeof result.finalOutput === "string" ? (result.finalOutput || "") : JSON.stringify(result.finalOutput));
      } else {
        console.error('\nExecution failed!');
        process.exit(1);
      }

      await runner.shutdown();
    } catch (error: any) {
      await runner.shutdown();
      process.stderr.write('Fatal error');
      process.exit(1);
    }
  });

program.parse();
