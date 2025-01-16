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
