# Cerebras Plugin for Dify

Official Cerebras model provider plugin for [Dify](https://dify.ai), enabling ultra-fast AI inference in your Dify applications.

## Quick Install

1. Download [`cerebras.difypkg`](./releases/cerebras.difypkg) from the releases folder
2. In Dify, go to **Settings** > **Plugins** > **Install Plugin**
3. Upload the `.difypkg` file
4. Configure your Cerebras API key (get one at [cloud.cerebras.ai](https://cloud.cerebras.ai))

## Available Models

| Model | Description |
|-------|-------------|
| `llama-3.3-70b` | Best for complex reasoning and long-form content |
| `qwen-3-32b` | Balanced performance for general-purpose tasks |
| `llama3.1-8b` | Fastest model, ideal for simple tasks and high throughput |
| `gpt-oss-120b` | Largest model for demanding tasks |
| `zai-glm-4.6` | Advanced 357B parameter model with strong reasoning |

## Building from Source

If you want to build the plugin yourself:

```bash
# Install the Dify plugin CLI
pip install dify-plugin-daemon

# Package the plugin
dify-plugin package ./
```

This will generate a new `cerebras.difypkg` file.

## Documentation

- [Full Integration Guide](https://inference-docs.cerebras.ai/integrations/dify)
- [Cerebras API Reference](https://inference-docs.cerebras.ai/api-reference)
- [Dify Documentation](https://docs.dify.ai)

## License

MIT
