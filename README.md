# AI Channel Assistant API

## Overview
This API service provides an intelligent channel assistant that can process and respond to messages using RAG (Retrieval-Augmented Generation) technology. It integrates with Pinecone for vector storage, OpenAI for embeddings and text generation, and Supabase for data persistence.

## Features
- Document processing (PDF, DOCX)
- Message history vectorization
- Intelligent response generation
- Real-time index updates
- Channel-based context management

## Prerequisites
- Python 3.11+
- Docker
- Supabase account
- Pinecone account
- OpenAI API key

## Environment Variables
Create a `.env` file with the following variables:

## Installation

### Using Docker (Recommended)

### Local Development

## API Endpoints

### 1. Update Index

## Monitoring
The application includes a health check endpoint at `/status` that can be used for monitoring the service's health.

## Error Handling
The API returns appropriate HTTP status codes:
- 200: Successful operation
- 400: Bad request
- 500: Internal server error

## Security Considerations
- All sensitive information is stored in environment variables
- Docker container runs as non-root user
- API keys should be kept secure and rotated regularly

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License

Copyright (c) 2024 Gauntlet AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support
For support, please open an issue in the GitHub repository or contact the development team.

## Authors
Gauntlet AI Team

## Acknowledgments
- OpenAI for providing the language model
- Pinecone for vector storage
- Supabase for database services

## Testing
