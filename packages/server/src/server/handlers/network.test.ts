import { createMockModel } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { Mastra } from '@mastra/core/mastra';
import { AgentNetwork } from '@mastra/core/network';
import { RuntimeContext } from '@mastra/core/runtime-context';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HTTPException } from '../http-exception';
import { getNetworksHandler, getNetworkByIdHandler, generateHandler, streamGenerateHandler } from './network';

function createMockAgent(name: string) {
  return new Agent({
    name,
    instructions: 'You are a helpful assistant',
    model: createMockModel({ mockText: 'Hello, world!' }),
  });
}

function createMockNetwork(name: string, agents: Agent[] = []) {
  return new AgentNetwork({
    name,
    instructions: 'You are a helpful assistant',
    agents,
    model: createMockModel({ mockText: 'Hello, world!' }),
  });
}

describe('Network Handlers', () => {
  let mockMastra: Mastra;
  let mockNetwork: AgentNetwork;
  let mockAgents: Agent[];

  const runtimeContext = new RuntimeContext();

  beforeEach(() => {
    vi.clearAllMocks();
    mockAgents = [createMockAgent('agent1'), createMockAgent('agent2')];
    mockNetwork = createMockNetwork('test-network', mockAgents);
    mockMastra = new Mastra({
      logger: false,
      networks: {
        'test-network': mockNetwork,
      },
    });
  });

  describe('getNetworksHandler', () => {
    it('should get all networks successfully', async () => {
      const result = await getNetworksHandler({ mastra: mockMastra, runtimeContext });

      expect(result).toEqual([
        {
          id: 'test-network',
          name: 'test-network',
          instructions: expect.any(String),
          agents: [
            { name: 'agent1', provider: 'mock-provider', modelId: 'mock-model-id' },
            { name: 'agent2', provider: 'mock-provider', modelId: 'mock-model-id' },
          ],
          routingModel: { provider: 'mock-provider', modelId: 'mock-model-id' },
        },
      ]);
    });
  });

  describe('getNetworkByIdHandler', () => {
    it('should throw error when networkId is not provided', async () => {
      await expect(getNetworkByIdHandler({ mastra: mockMastra, runtimeContext })).rejects.toThrow('Network not found');
    });

    it('should throw error when network is not found', async () => {
      await expect(
        getNetworkByIdHandler({ mastra: mockMastra, runtimeContext, networkId: 'non-existent' }),
      ).rejects.toThrow('Network not found');
    });

    it('should get network by ID successfully', async () => {
      const result = await getNetworkByIdHandler({
        mastra: mockMastra,
        runtimeContext,
        networkId: 'test-network',
      });

      expect(result).toEqual({
        id: 'test-network',
        name: 'test-network',
        instructions: expect.any(String),
        agents: [
          { name: 'agent1', provider: 'mock-provider', modelId: 'mock-model-id' },
          { name: 'agent2', provider: 'mock-provider', modelId: 'mock-model-id' },
        ],
        routingModel: { provider: 'mock-provider', modelId: 'mock-model-id' },
      });
    });
  });

  describe('generateHandler', () => {
    it('should throw error when networkId is not provided', async () => {
      await expect(
        generateHandler({
          mastra: mockMastra,
          body: {
            messages: ['test message'],
            resourceId: 'test-resource',
            threadId: 'test-thread',
            experimental_output: undefined,
          },
          runtimeContext: new RuntimeContext(),
        }),
      ).rejects.toThrow(new HTTPException(404, { message: 'Network not found' }));
    });

    it('should throw error when network is not found', async () => {
      await expect(
        generateHandler({
          mastra: mockMastra,
          networkId: 'non-existent',
          body: {
            messages: ['test message'],
            resourceId: 'test-resource',
            threadId: 'test-thread',
            experimental_output: undefined,
          },
          runtimeContext: new RuntimeContext(),
        }),
      ).rejects.toThrow(new HTTPException(404, { message: 'Network not found' }));
    });

    it('should throw error when messages are not provided', async () => {
      await expect(
        generateHandler({
          mastra: mockMastra,
          networkId: 'test-network',
          body: {
            resourceId: 'test-resource',
            threadId: 'test-thread',
            experimental_output: undefined,
          },
          runtimeContext: new RuntimeContext(),
        }),
      ).rejects.toThrow(new HTTPException(400, { message: 'Argument "messages" is required' }));
    });

    it('should generate successfully', async () => {
      const mockMessages = ['test message'];
      const mockResult = { text: 'generated response' } as any;
      vi.spyOn(mockNetwork, 'generate').mockResolvedValue(mockResult);

      const result = await generateHandler({
        mastra: mockMastra,
        networkId: 'test-network',
        body: {
          messages: mockMessages,
          resourceId: 'test-resource',
          threadId: 'test-thread',
          experimental_output: undefined,
        },
        runtimeContext: new RuntimeContext(),
      });

      expect(result).toEqual(mockResult);
    });

    it('should merge runtime context from request body', async () => {
      const generateSpy = vi.spyOn(mockNetwork, 'generate').mockResolvedValue({} as any);
      const rc = new RuntimeContext();
      rc.set('a', 1);

      await generateHandler({
        mastra: mockMastra,
        networkId: 'test-network',
        body: {
          messages: ['hi'],
          resourceId: 'r',
          threadId: 't',
          experimental_output: undefined,
          runtimeContext: { b: 2 },
        },
        runtimeContext: rc,
      });

      const passed = generateSpy.mock.calls[0][1].runtimeContext as RuntimeContext;
      expect(passed.get('a')).toBe(1);
      expect(passed.get('b')).toBe(2);
    });
  });

  describe('streamGenerateHandler', () => {
    it('should throw error when networkId is not provided', async () => {
      await expect(
        streamGenerateHandler({
          mastra: mockMastra,
          body: {
            messages: ['test message'],
            resourceId: 'test-resource',
            threadId: 'test-thread',
            experimental_output: undefined,
          },
          runtimeContext: new RuntimeContext(),
        }),
      ).rejects.toThrow(new HTTPException(404, { message: 'Network not found' }));
    });

    it('should throw error when network is not found', async () => {
      await expect(
        streamGenerateHandler({
          mastra: mockMastra,
          networkId: 'non-existent',
          body: {
            messages: ['test message'],
            resourceId: 'test-resource',
            threadId: 'test-thread',
            experimental_output: undefined,
          },
          runtimeContext: new RuntimeContext(),
        }),
      ).rejects.toThrow(new HTTPException(404, { message: 'Network not found' }));
    });

    it('should throw error when messages are not provided', async () => {
      await expect(
        streamGenerateHandler({
          mastra: mockMastra,
          networkId: 'test-network',
          body: {
            resourceId: 'test-resource',
            threadId: 'test-thread',
            experimental_output: undefined,
          },
          runtimeContext: new RuntimeContext(),
        }),
      ).rejects.toThrow(new HTTPException(400, { message: 'Argument "messages" is required' }));
    });

    it('should stream generate successfully', async () => {
      const mockMessages = ['test message'];
      const mockStreamResult = {
        [Symbol.asyncIterator]: async function* () {
          yield { text: 'streamed response' };
        },
      } as any;
      const mockStream = {
        toDataStreamResponse: vi.fn().mockReturnValue(mockStreamResult),
      };

      vi.spyOn(mockNetwork, 'stream').mockResolvedValue(mockStream as any);

      const result = await streamGenerateHandler({
        mastra: mockMastra,
        networkId: 'test-network',
        body: {
          messages: mockMessages,
          resourceId: 'test-resource',
          threadId: 'test-thread',
          experimental_output: undefined,
        },
        runtimeContext: new RuntimeContext(),
      });

      expect(result).toEqual(mockStreamResult);
    });

    it('should merge runtime context from request body when streaming', async () => {
      const streamSpy = vi.spyOn(mockNetwork, 'stream').mockResolvedValue({
        toDataStreamResponse: vi.fn(),
      } as any);

      const rc = new RuntimeContext();
      rc.set('x', 'y');

      const mockMessages = ['test message'];

      await streamGenerateHandler({
        mastra: mockMastra,
        networkId: 'test-network',
        body: {
          messages: mockMessages,
          resourceId: 'r',
          threadId: 't',
          experimental_output: undefined,
          runtimeContext: { foo: 'bar' },
        },
        runtimeContext: rc,
      });

      const passed = streamSpy.mock.calls[0][1].runtimeContext as RuntimeContext;
      expect(passed.get('x')).toBe('y');
      expect(passed.get('foo')).toBe('bar');
    });
  });
});
