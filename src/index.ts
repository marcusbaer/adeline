
import { AgentsRunner, type AgentsRunnerContext } from './agents/agents-runner.js';
import { randomUUID } from 'node:crypto';

if (process.env.DEBUG === 'false') {
  console.debug = () => {};
}

async function main(context: AgentsRunnerContext) {
  const runner = new AgentsRunner();

  try {
    await runner.initialize();

    const executionId = `exec_${Date.now()}_${randomUUID().split('-')[0]}`;
    const result = await runner.runJob(context);

    if (result.status === 'completed') {
      console.debug('\n✅ Completed successfully!');
    } else {
      console.error('\nExecution failed!');
      console.error(`   Phase: ${result.error?.phase}`);
      console.error(`   Error: ${result.error?.message}`);
      process.exit(1);
    }

    await runner.shutdown();

    return result;

  } catch (error: any) {
    console.error('\nFatal error:', error.message);
    await runner.shutdown();

    throw Error('Fatal error');
  }
}

const uuid = randomUUID();

main({
  executionId: uuid as string,
  filePath: "shared/page.html",
}).then((result) => {
  console.log(result.finalOutput || "");
}).catch((error) => {
  console.error(error);
});
